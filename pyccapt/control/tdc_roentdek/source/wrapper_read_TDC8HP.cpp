//////////////////////////////////////////////////////////////////////////////////////////
//
//	written by Achim Czasch (RoentDek Handels Gmbh)
//
//////////////////////////////////////////////////////////////////////////////////////////


#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
using namespace std;
using namespace std::chrono;

#include "tdcmanager.h"
#include <windows.h> // Sleep, GetTickCount64 (MFC/afx.h is not needed)
#include "conio.h"

#include "LMF_IO.h"
#define NUM_CHANNELS (80)
#define NUM_IONS (32)


#define READ_BUFFER_SIZE (50000)
// this size can be set to other values

// TDC channel layout (0-based): 0..5 = u1,u2,v1,v2,w1,w2, 6 = start (laser), 7 = MCP.
// The MCP is the group trigger, so every time in a group is relative to the MCP (t_mcp = 0).
#define START_CHANNEL (6)
#define MCP_CHANNEL (7)
// Event row: 12 first-hit values (layout of DLL builds before 2026-11), then
// the hit count of each channel and up to HITS_PER_CHANNEL hits per channel,
// so Python can recover ions whose first hit is a reflection or noise pulse.
#define FIRST_HIT_VALUES (12)
#define HITS_PER_CHANNEL (4)
#define VALUES_PER_EVENT (FIRST_HIT_VALUES + 8 + 8 * HITS_PER_CHANNEL)


// Hexanode constants, same values and meaning as DAn-part2.ccf (Parameter 1102..1120).
// pyccapt.calibration.data_tools.roentdek_hexanode implements the same reconstruction;
// the Python acquisition loop recomputes x/y/t from the raw channels with it.
struct HexParams
{
	double fu = 0.890;			// mm/ns, layer u
	double fv = 0.917;			// mm/ns, layer v
	double fw = 0.887;			// mm/ns, layer w
	double w_offset = -1.175;	// ns
	double x_offset = -1.5;		// mm
	double y_offset = 1.85;		// mm
	double sum_offset[3] = { -113.98, -113.03, -114.85 };	// ns, shifts each time sum to zero
	double sum_halfwidth[3] = { 1.5, 1.2, 2.3 };				// ns, time-sum gate
	double runtime[3] = { 88., 86., 88. };					// ns, max signal runtime across a layer
	double rotation_deg = 0.;
	double rotation_center[2] = { 0., 0. };					// mm
	double tof_offset = 0.;									// ns
};

#ifdef WIN64
	#pragma comment(lib,"hptdc_driver_3.5.2_x64.lib")
#else
	#pragma comment(lib,"hptdc_driver_3.5.2_x86.lib")
#endif





bool PCIGetTDC_25psGroupMode_TDC8HP(TDCManager * manager, unsigned __int32  CNT[], __int32  TDC[][10], unsigned __int32 Max_channels, unsigned __int32 Max_hits, unsigned __int64 &ui64_HPTDC_AbsoluteTimeStamp);


class Wrraper_tdc
{
public:
	Wrraper_tdc(int, int);
	int init_tdc();
	int run_tdc();
	int stop_tdc();
	double* get_data_tdc();
	double* get_data_tdc_buf();


private:

	void reconstruct_hit(double& x, double& y, double& tof);
	void store_event(double* row);

	// now configure the TDC:
	TDCManager* manager = new TDCManager(0x1A13, 0x0001);
	LMF_IO* LMF = new LMF_IO(NUM_CHANNELS, NUM_IONS);

	int buffer_size;
	int time_out_micro;

	__int32 trigger_channel;


	unsigned __int64 ui64_HPTDC_AbsoluteTimeStamp;
	// note: please do not confuse binsize with resolution. The real (internal) binsize of each channel is 25 ps.
	//       Therefore the resolution of a measured time (e.g. channel 1 vs. trigger channel) is sqrt(2)*25ps = 34ps FWHM = 14.5 ps RMS.

	__int64 recorded_triggers = 0;
	__int64 last_tick_ms = GetTickCount64();



	//  now read events from TDC
	unsigned __int32	CNT[8];
	__int32				TDC_data[8][10];

	HexParams hex_params;

	double X;
	double Y;
	double TOF;

	double* array_tem;

	double tdc_binsize;

};

Wrraper_tdc::Wrraper_tdc(int buf_size, int time_out)
{
	buffer_size = buf_size;
	time_out_micro = time_out * 1000; // convert it from milisecond to microsecond
	trigger_channel = 8 - 1; // 1st TDC channel has index 0
	ui64_HPTDC_AbsoluteTimeStamp = 0;
	tdc_binsize = 0.025;	// overwritten in init_tdc() with the value reported by the driver
	memset(CNT, 0, sizeof(CNT));
	memset(TDC_data, 0, sizeof(TDC_data));

	X = 0;
	Y = 0;
	TOF = 0;

	array_tem = new double[(VALUES_PER_EVENT * buffer_size) + 1]; //+1 for putting the size of array
	array_tem[0] = 0;

};


// Hexanode position and TOF of the first hit on each channel, in the DAn convention:
//   u = 0.5*fu*(u1-u2), v = 0.5*fv*(v1-v2), w = 0.5*fw*(w1-w2+w_offset)
//   Xuv = u, Yuv = (u-2v)/sqrt(3); Xuw = u, Yuw = (2w-u)/sqrt(3); Xvw = v+w, Yvw = (w-v)/sqrt(3)
// A layer is used only if its time sum is inside the gate. One missing signal of a layer is
// rebuilt from the time sum when exactly one other layer is complete. x, y and tof are NaN
// when no position can be built or the start channel did not fire (DAn reports no particle).
void Wrraper_tdc::reconstruct_hit(double& x, double& y, double& tof)
{
	const double nan = numeric_limits<double>::quiet_NaN();
	x = nan;
	y = nan;
	tof = nan;
	if (CNT[START_CHANNEL] == 0) return;

	const double t_mcp = 0.;	// MCP is the group trigger
	tof = t_mcp - TDC_data[START_CHANNEL][0] * tdc_binsize + hex_params.tof_offset;

	const double factor[3] = { hex_params.fu, hex_params.fv, hex_params.fw };
	const double layer_offset[3] = { 0., 0., hex_params.w_offset };
	double pos[3];
	bool full[3];
	bool rebuilt[3];
	int n_full = 0;
	for (int k = 0; k < 3; ++k) {
		const bool p1 = CNT[2 * k] > 0;
		const bool p2 = CNT[2 * k + 1] > 0;
		double t1 = TDC_data[2 * k][0] * tdc_binsize;
		double t2 = TDC_data[2 * k + 1][0] * tdc_binsize;
		const double target = 2. * t_mcp - hex_params.sum_offset[k];
		full[k] = false;
		rebuilt[k] = false;
		if (p1 && p2) {
			full[k] = fabs(t1 + t2 - 2. * t_mcp + hex_params.sum_offset[k]) <= hex_params.sum_halfwidth[k];
		}
		else if (p1) {
			t2 = target - t1;
			rebuilt[k] = true;
		}
		else if (p2) {
			t1 = target - t2;
			rebuilt[k] = true;
		}
		const double diff = t1 - t2;
		const bool in_runtime = fabs(diff) <= hex_params.runtime[k] + hex_params.sum_halfwidth[k];
		full[k] = full[k] && in_runtime;
		rebuilt[k] = rebuilt[k] && in_runtime;
		pos[k] = 0.5 * factor[k] * (diff + layer_offset[k]);
		if (full[k]) ++n_full;
	}

	int mask = 0;	// u = 1, v = 2, w = 4
	for (int k = 0; k < 3; ++k) {
		if (full[k] || (rebuilt[k] && n_full == 1)) mask |= (1 << k);
	}
	const double u = pos[0];
	const double v = pos[1];
	const double w = pos[2];
	const double sqrt3 = sqrt(3.);
	double px;
	double py;
	switch (mask) {
	case 7: px = (2. * u + v + w) / 3.; py = (w - v) / sqrt3; break;
	case 3: px = u; py = (u - 2. * v) / sqrt3; break;
	case 5: px = u; py = (2. * w - u) / sqrt3; break;
	case 6: px = v + w; py = (w - v) / sqrt3; break;
	default: return;
	}
	px += hex_params.x_offset;
	py += hex_params.y_offset;
	if (hex_params.rotation_deg != 0.) {
		const double angle = hex_params.rotation_deg * 3.14159265358979323846 / 180.;
		const double dx = px - hex_params.rotation_center[0];
		const double dy = py - hex_params.rotation_center[1];
		px = hex_params.rotation_center[0] + dx * cos(angle) - dy * sin(angle);
		py = hex_params.rotation_center[1] + dx * sin(angle) + dy * cos(angle);
	}
	x = px;
	y = py;
}


// One event row (VALUES_PER_EVENT values):
//   [0..7]   ch0..ch7 first-hit bins (0 = no hit)
//   [8..11]  X [mm], Y [mm], TOF [ns] of the first-hit reconstruction, timestamp [bins]
//   [12..19] number of hits per channel (as counted by the driver, up to 10)
//   [20..]   HITS_PER_CHANNEL hits per channel, channel-major (0 = unused slot)
void Wrraper_tdc::store_event(double* row)
{
	reconstruct_hit(X, Y, TOF);
	for (int i = 0; i < 8; ++i) {
		row[i] = CNT[i] > 0 ? static_cast<double>(TDC_data[i][0]) : 0.;
	}
	row[8] = X;
	row[9] = Y;
	row[10] = TOF;
	row[11] = static_cast<double>(ui64_HPTDC_AbsoluteTimeStamp);
	for (int i = 0; i < 8; ++i) {
		row[FIRST_HIT_VALUES + i] = static_cast<double>(CNT[i]);
		for (int h = 0; h < HITS_PER_CHANNEL; ++h) {
			row[FIRST_HIT_VALUES + 8 + HITS_PER_CHANNEL * i + h] =
				h < static_cast<int>(CNT[i]) ? static_cast<double>(TDC_data[i][h]) : 0.;
		}
	}
}



int Wrraper_tdc::init_tdc()
{

	//  --------------------------------------------------------
	//	read Header information of template LMF-file (ignore this part)
	char template_name[270] = "simple_read_TDC8HP";


	while (strlen(template_name) > 0) {
		if (template_name[strlen(template_name) - 1] == '\\' || template_name[strlen(template_name) - 1] == '/') {
			break;
		}
		template_name[strlen(template_name) - 1] = 0;
	}

	sprintf(template_name + strlen(template_name), "%s", "CoboldPC2011R5-2_Header-Template_1TDC8HP.lmf");

	if (!LMF->OpenInputLMF(template_name)) {
		printf("error: could not open the template file\n%s\n", template_name);
		printf("It must be in the same folder as the exe.\n");
		return 1;
	}
	LMF->CloseInputLMF();
	//  --------------------------------------------------------

	if (!manager) return -1;

	manager->Init();

	TDCInfo myTDC;
	myTDC = manager->getTDCInfo(0);
	int Firmware = (myTDC.version & 0xff);
	if (Firmware < 6) printf("Please update the firmware of the TDC.\n");


	tdc_binsize = myTDC.resolution * 1.e9;
	double timestamp_binsize = tdc_binsize;


	// note: please do not confuse binsize with resolution. The real (internal) binsize of each channel is 25 ps.
	//       Therefore the resolution of a measured time (e.g. channel 1 vs. trigger channel) is sqrt(2)*25ps = 34ps FWHM = 14.5 ps RMS. 
	
	// now reconfigure the TDC (necessary after all parameters are set):
	manager->Reconfigure();

	LMF->TDC8HP.GroupRangeEnd_p70 = atof(manager->GetParameter("GroupRangeEnd"));
	LMF->TDC8HP.GroupRangeStart_p69 = atof(manager->GetParameter("GroupRangeStart"));
	LMF->TDC8HP.TriggerDeadTime_p68 = atof(manager->GetParameter("Triggerdeadtime"));
	LMF->TDC8HP.TriggerChannel_p64 = atoi(manager->GetParameter("triggerchannel"));

	LMF->tdcresolution_output = myTDC.resolution * 1e9;

	//	Open and prepare output file
	LMF->number_of_channels_output = 8;
	LMF->max_number_of_hits_output = 10;
	LMF->frequency = 1.;
	LMF->OpenOutputLMF("output.lmf");

	return 0;
};

int Wrraper_tdc::run_tdc()
{
	manager->Start();
	return 0;
};

int Wrraper_tdc::stop_tdc()
{
	// stop TDC
	printf("stopping the TDC.\n");
	manager->Stop();
	manager->ClearBuffer();

	manager->CleanUp(); // cleanup is necessary before delete.
	delete (TDCManager*)manager;

	if (LMF) {
		LMF->CloseOutputLMF();
		delete LMF;
	}

	return 0;
};


double* Wrraper_tdc::get_data_tdc()
{

	memset(TDC_data, 0, sizeof(TDC_data));	// clear TDC_data[] array

	bool group_is_complete = false;
	while (!group_is_complete) {
		group_is_complete = PCIGetTDC_25psGroupMode_TDC8HP(manager, CNT, TDC_data, 8, 10, ui64_HPTDC_AbsoluteTimeStamp);
		if (!group_is_complete) {
			Sleep(1);
		}
	}

	LMF->WriteTDCData(ui64_HPTDC_AbsoluteTimeStamp, CNT, &TDC_data[0][0]);

	array_tem[0] = 1;
	store_event(array_tem + 1);
	return array_tem;
}


// Version 2 of get_data_tdc with buffering the data to reduce returning number to python
// It buffers events and then return the result to Python wrraper.
// array_tem[0] = number of events in this packet (0 on timeout), then VALUES_PER_EVENT values per event.
// The count restarts at 0 on every call; the timeout is also checked while waiting for a group,
// so the call returns within ~time_out ms even when no ion arrives and Python can stop the TDC.
double* Wrraper_tdc::get_data_tdc_buf()
{
	__int64 packet_events = 0;
	array_tem[0] = 0;

	auto start = high_resolution_clock::now();

	while (packet_events < buffer_size) {

		memset(TDC_data, 0, sizeof(TDC_data));	// clear TDC_data[] array

		bool group_is_complete = PCIGetTDC_25psGroupMode_TDC8HP(manager, CNT, TDC_data, 8, 10, ui64_HPTDC_AbsoluteTimeStamp);
		if (group_is_complete) {
			LMF->WriteTDCData(ui64_HPTDC_AbsoluteTimeStamp, CNT, &TDC_data[0][0]);
			store_event(array_tem + 1 + VALUES_PER_EVENT * packet_events);
			++packet_events;
			++recorded_triggers;
			array_tem[0] = static_cast<double>(packet_events);
		}

		auto duration = duration_cast<microseconds>(high_resolution_clock::now() - start);
		if (duration.count() >= time_out_micro) {
			break;
		}
		if (!group_is_complete) {
			Sleep(1);
		}
	}
	return array_tem;
}


// Define C functions for the C++ class - as ctypes can only talk to C...
extern "C"
{
	__declspec(dllexport) Wrraper_tdc* Warraper_tdc_new(int buf_size, int time_out) { return new Wrraper_tdc(buf_size, time_out); }
	__declspec(dllexport) int init_tdc(Wrraper_tdc* Wrraper_tdc) { return Wrraper_tdc->init_tdc(); }
	__declspec(dllexport) int run_tdc(Wrraper_tdc* Wrraper_tdc) { return Wrraper_tdc->run_tdc(); }
	__declspec(dllexport) int stop_tdc(Wrraper_tdc* Wrraper_tdc) { return Wrraper_tdc->stop_tdc(); }
	__declspec(dllexport) double* get_data_tdc(Wrraper_tdc* Wrraper_tdc) { return Wrraper_tdc->get_data_tdc(); }
	__declspec(dllexport) double* get_data_tdc_buf(Wrraper_tdc* Wrraper_tdc) { return Wrraper_tdc->get_data_tdc_buf(); }
	// Packet layout, so Python can tell this build from the 12-value first-hit builds.
	__declspec(dllexport) int values_per_event() { return VALUES_PER_EVENT; }
	__declspec(dllexport) int hits_per_channel() { return HITS_PER_CHANNEL; }
}


//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[], char* envp[])
//////////////////////////////////////////////////////////////////////////////////////////
{
	return 0;
}



/////////////////////////////////////////////////////////////////// 
// 25ps GroupMode:
// In this mode the grouping is done inside the TDC driver
///////////////////////////////////////////////////////////////////
bool PCIGetTDC_25psGroupMode_TDC8HP(TDCManager * manager, unsigned __int32 CNT[], __int32 TDC[][10], unsigned __int32 Max_channels, unsigned __int32 Max_hits, unsigned __int64 &ui64_HPTDC_AbsoluteTimeStamp)
{
	HIT buffer[READ_BUFFER_SIZE];

	memset(CNT,0,sizeof(__int32)*Max_channels);

	static unsigned __int32 ulRollOver = 0;
	static unsigned __int32 ulAbsoluteTimeStamp = 0;
	static unsigned __int64	ui64_OldTimeStamp = 0;
	
	if (!ui64_HPTDC_AbsoluteTimeStamp) {
		ulRollOver = 0;
		ulAbsoluteTimeStamp = 0;
		ui64_OldTimeStamp = 0;
	}

	//------------------------------------------
	// Read one Event, count = number of hits
	//------------------------------------------
	unsigned __int32 count = manager->Read(buffer, READ_BUFFER_SIZE);
	if(!count) return false;
	//------------------------------------------

	bool bOKFlag = false;
	// now process the data
	for(unsigned __int32 i = 0; i < count ; ++i)
	{
		unsigned __int32 uldata_word = buffer[i];

		if( (uldata_word&0xC0000000) > 0x40000000)		// valid data only if rising or falling transition
		{
			long lTDCData = (uldata_word&0x00FFFFFF);
			if(lTDCData & 0x00800000) lTDCData |= 0xff000000;	// detect 24 bit signed flag:  if detected extend negative value to 32 bit

			unsigned char ucTDCChannel = (unsigned char)((uldata_word&0x3F000000)>>24);	// extract channel information
			
			if(ucTDCChannel < Max_channels)	// if detected channel fits into TDC array ...
			{
				// increase Hit Counter;
				++CNT[ucTDCChannel];

				// test for oversized Hits
				if(CNT[ucTDCChannel] > Max_hits) {
					--CNT[ucTDCChannel];
				} else {
					// if Hit # ok then store it
					TDC[ucTDCChannel][CNT[ucTDCChannel]-1] = lTDCData;
					bOKFlag = true;
				}
			}
		} else {
			if (uldata_word >> 24 == 0x10) {
				ulRollOver = uldata_word & 0xffffff; // RollOverWord detected ?
			}
			if (!(uldata_word & 0xf0000000)) ulAbsoluteTimeStamp = uldata_word & 0xffffff; // GroupWord detected
		}
	}	

	if (count) {
		unsigned __int64 ui64_TimeStamp_temp = ulRollOver * 0x1000000ull + ulAbsoluteTimeStamp;

		// handle 48 bit rollovers:
		ui64_HPTDC_AbsoluteTimeStamp += ui64_TimeStamp_temp - ui64_OldTimeStamp;
		if (ui64_TimeStamp_temp < ui64_OldTimeStamp) {
			ui64_HPTDC_AbsoluteTimeStamp += 0x1000000000000ull; 
		}

		ui64_OldTimeStamp = ui64_TimeStamp_temp;
	}

	return bOKFlag;
}


