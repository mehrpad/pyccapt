//////////////////////////////////////////////////////////////////////////////////////////
//
//  RoentDek TDC8HP wrapper for PyCCAPT  -  version 3 (2026-11)
//
//  based on simple_read_TDC8HP.cpp written by Achim Czasch (RoentDek Handels GmbH)
//  modified by Mehrpad Monajem
//
//  What this DLL does
//  ------------------
//  * configures the TDC exactly like before (global.cfg in the working directory,
//    TDCManager::Init + Reconfigure) and writes EVERY group to "output.lmf"
//    exactly like before (same Cobold template header, 8 channels x 10 hits,
//    raw TDC bins, 64 bit time stamp). The .lmf file can always be evaluated
//    with CoboldPC.
//  * get_data_tdc_raw():  returns ALL hits of every group (not only the first
//    hit) to Python:  time stamp [bins], cnt[8], tdc[8][10] [bins].
//    The call returns after max_events groups or after timeout_ms, also when
//    no event arrives (the old DLL could block forever).
//  * get_data_tdc_buf():  old interface (12 doubles per event) kept for
//    compatibility, with the bugs of the old versions fixed.
//  * diagnostics (version 3): counters of driver reads / data words, the last
//    raw words and the text of driver exceptions -> get_diagnostics(),
//    get_last_words(), get_last_error().
//
//  Changes compared to the old DLL
//  -------------------------------
//  * LMF_IO is created with (8 channels, 10 hits) = the size of TDC_data[8][10].
//    (verified: the written .lmf is byte-identical to the old DLL's files)
//  * the time stamp / roll-over state is per object (not reset between calls)
//  * time-out works also when no data arrive -> stop_tdc() is always reached
//  * exceptions of the RoentDek driver are caught and reported
//  * no MFC (afx.h) needed
//
//  Build: see build_wrapper.bat  (MOCK_TDC builds a test version that replays a
//  recorded run instead of talking to the hardware)
//
//////////////////////////////////////////////////////////////////////////////////////////

#ifndef WIN64
#define WIN64
#endif
#ifndef WIN32
#define WIN32
#endif

#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

#ifdef MOCK_TDC
#include "mock_tdcmanager.h"
#else
#include "tdcmanager.h"
#pragma comment(lib, "hptdc_driver_3.5.2_x64.lib")
#endif

#include "LMF_IO.h"

#define NUM_TDC_CHANNELS (8)
#define NUM_TDC_HITS (10)
#define READ_BUFFER_SIZE (50000)
#define N_LAST_WORDS (64)

#define WRAPPER_VERSION (3)

static const char* LMF_TEMPLATE_NAME = "CoboldPC2011R5-2_Header-Template_1TDC8HP.lmf";
static const char* LMF_OUTPUT_NAME = "output.lmf";

// diagnostic counters (index in get_diagnostics)
enum {
	D_READ_CALLS = 0,      // calls of TDCManager::Read
	D_READ_NONZERO,        // calls that returned > 0 words
	D_READ_NEGATIVE,       // calls that returned < 0
	D_WORDS,               // all words
	D_WORDS_HIT,           // rising/falling edge data words
	D_WORDS_HIT_BAD_CH,    // data words with channel >= 8
	D_WORDS_HIT_OVERFLOW,  // data words dropped (> 10 hits in a channel)
	D_WORDS_GROUP,         // group words (time stamp)
	D_WORDS_ROLLOVER,      // roll-over words
	D_WORDS_OTHER,         // everything else (error words, ...)
	D_GROUPS_OK,           // groups with at least one valid hit (written to LMF / returned)
	D_GROUPS_EMPTY,        // non-empty reads without a valid hit
	D_EXCEPTIONS,          // exceptions thrown by the driver
	D_LMF_EVENTS,          // events written to output.lmf
	D_STATE,               // 0 created, 1 initialized, 2 running, 3 stopped, -1 init failed
	D_N
};

class Wrraper_tdc
{
public:
	Wrraper_tdc(int buf_size, int time_out_ms);
	~Wrraper_tdc();
	int init_tdc();
	int run_tdc();
	int stop_tdc();
	int read_events(unsigned __int64* ts_out, unsigned __int32* cnt_out, __int32* tdc_out, int max_events, int timeout_ms);
	double* get_data_tdc_buf();
	double get_bin_size_ns() { return tdc_binsize; }
	__int64 get_number_written_events() { return diag[D_LMF_EVENTS]; }
	int get_diagnostics(__int64* out, int n) {
		int m = (n < D_N) ? n : D_N;
		for (int i = 0; i < m; ++i) out[i] = diag[i];
		return m;
	}
	int get_last_words(unsigned __int32* out, int n) {
		int m = (n < n_last_words) ? n : n_last_words;
		memcpy(out, last_words, sizeof(unsigned __int32) * m);
		return m;
	}
	const char* get_last_error() { return last_error; }
	int get_tdc_info(double* out, int n) {
		if (n < 4) return 0;
		out[0] = info_resolution; out[1] = info_version; out[2] = info_serial; out[3] = info_channels;
		return 4;
	}

private:
	bool read_group();   // PCIGetTDC_25psGroupMode_TDC8HP
	void set_error(const char* where, const char* what) {
		_snprintf(last_error, sizeof(last_error) - 1, "%s: %s", where, what ? what : "");
		last_error[sizeof(last_error) - 1] = 0;
		printf("RoentDek wrapper error - %s\n", last_error);
		fflush(stdout);
	}

	TDCManager* manager;
	LMF_IO* LMF;
	bool initialized;
	bool running;
	bool stopped;

	int buffer_size;
	int time_out_ms;

	HIT* hit_buffer;

	// current group
	unsigned __int32 CNT[NUM_TDC_CHANNELS];
	__int32 TDC_data[NUM_TDC_CHANNELS][NUM_TDC_HITS];

	// time stamp state (kept between calls)
	unsigned __int64 ui64_HPTDC_AbsoluteTimeStamp;
	unsigned __int32 ulRollOver;
	unsigned __int32 ulAbsoluteTimeStamp;
	unsigned __int64 ui64_OldTimeStamp;

	double tdc_binsize;      // ns
	double info_resolution, info_version, info_serial, info_channels;

	__int64 diag[D_N];
	unsigned __int32 last_words[N_LAST_WORDS];
	int n_last_words;
	char last_error[512];

	// buffers for the legacy interface
	double* legacy_array;
	unsigned __int64* legacy_ts;
	unsigned __int32* legacy_cnt;
	__int32* legacy_tdc;
};


Wrraper_tdc::Wrraper_tdc(int buf_size, int time_out)
{
	buffer_size = (buf_size > 0) ? buf_size : 1;
	time_out_ms = (time_out > 0) ? time_out : 1;
	manager = NULL;
	LMF = NULL;
	initialized = false;
	running = false;
	stopped = false;
	hit_buffer = new HIT[READ_BUFFER_SIZE];
	memset(CNT, 0, sizeof(CNT));
	memset(TDC_data, 0, sizeof(TDC_data));
	ui64_HPTDC_AbsoluteTimeStamp = 0;
	ulRollOver = 0;
	ulAbsoluteTimeStamp = 0;
	ui64_OldTimeStamp = 0;
	tdc_binsize = 0.025;
	info_resolution = info_version = info_serial = info_channels = 0;
	memset(diag, 0, sizeof(diag));
	memset(last_words, 0, sizeof(last_words));
	n_last_words = 0;
	last_error[0] = 0;
	legacy_array = new double[12 * (size_t)buffer_size + 1];
	memset(legacy_array, 0, sizeof(double) * (12 * (size_t)buffer_size + 1));
	legacy_ts = new unsigned __int64[buffer_size];
	legacy_cnt = new unsigned __int32[(size_t)buffer_size * NUM_TDC_CHANNELS];
	legacy_tdc = new __int32[(size_t)buffer_size * NUM_TDC_CHANNELS * NUM_TDC_HITS];
}

Wrraper_tdc::~Wrraper_tdc()
{
	stop_tdc();
	delete[] hit_buffer;
	delete[] legacy_array;
	delete[] legacy_ts;
	delete[] legacy_cnt;
	delete[] legacy_tdc;
}


int Wrraper_tdc::init_tdc()
{
	//  read header information of the template LMF file (same as the RoentDek example)
	LMF = new LMF_IO(NUM_TDC_CHANNELS, NUM_TDC_HITS);
	if (!LMF->OpenInputLMF((char*)LMF_TEMPLATE_NAME)) {
		set_error("init_tdc", "could not open the LMF template file (must be in the working directory)");
		delete LMF;
		LMF = NULL;
		diag[D_STATE] = -1;
		return 1;
	}
	LMF->CloseInputLMF();

	TDCInfo myTDC;
	try {
		// same sequence as the old DLL / the RoentDek example
		manager = new TDCManager(0x1A13, 0x0001);
		if (!manager) { diag[D_STATE] = -1; return -1; }
		manager->Init();
		myTDC = manager->getTDCInfo(0);
	}
	catch (TDCConfigException& e) {
		++diag[D_EXCEPTIONS]; set_error("TDCManager::Init", e.errorString); diag[D_STATE] = -1; return 3;
	}
	catch (...) {
		++diag[D_EXCEPTIONS]; set_error("TDCManager::Init", "unknown exception"); diag[D_STATE] = -1; return 3;
	}
	int Firmware = (myTDC.version & 0xff);
	if (Firmware < 6) printf("Please update the firmware of the TDC.\n");
	tdc_binsize = myTDC.resolution * 1.e9;
	info_resolution = myTDC.resolution;
	info_version = myTDC.version;
#ifndef MOCK_TDC
	info_serial = (double)myTDC.serialNumber;
	info_channels = myTDC.channelCount;
#endif

	// note: please do not confuse binsize with resolution. The real (internal) binsize of each channel is 25 ps.
	// now reconfigure the TDC (necessary after all parameters are set):
	try {
		manager->Reconfigure();
		LMF->TDC8HP.GroupRangeEnd_p70 = atof(manager->GetParameter("GroupRangeEnd"));
		LMF->TDC8HP.GroupRangeStart_p69 = atof(manager->GetParameter("GroupRangeStart"));
		LMF->TDC8HP.TriggerDeadTime_p68 = atof(manager->GetParameter("Triggerdeadtime"));
		LMF->TDC8HP.TriggerChannel_p64 = atoi(manager->GetParameter("triggerchannel"));
	}
	catch (TDCConfigException& e) {
		++diag[D_EXCEPTIONS]; set_error("TDCManager::Reconfigure", e.errorString); diag[D_STATE] = -1; return 4;
	}
	catch (...) {
		++diag[D_EXCEPTIONS]; set_error("TDCManager::Reconfigure", "unknown exception"); diag[D_STATE] = -1; return 4;
	}

	LMF->tdcresolution_output = myTDC.resolution * 1e9;

	//	open and prepare the output file (identical to the old DLL)
	LMF->number_of_channels_output = NUM_TDC_CHANNELS;
	LMF->max_number_of_hits_output = NUM_TDC_HITS;
	// LMF->frequency is NOT set: the old DLL kept the value read from the template
	// header (time stamp frequency used by CoboldPC), so the header stays identical.
	if (!LMF->OpenOutputLMF((char*)LMF_OUTPUT_NAME)) {
		set_error("init_tdc", "could not open output.lmf for writing");
		diag[D_STATE] = -1;
		return 2;
	}
	initialized = true;
	diag[D_STATE] = 1;
	return 0;
}

int Wrraper_tdc::run_tdc()
{
	if (!initialized || !manager) { set_error("run_tdc", "TDC not initialized"); return -1; }
	try {
		manager->Start();
	}
	catch (TDCConfigException& e) {
		++diag[D_EXCEPTIONS]; set_error("TDCManager::Start", e.errorString); return -2;
	}
	catch (...) {
		++diag[D_EXCEPTIONS]; set_error("TDCManager::Start", "unknown exception"); return -2;
	}
	running = true;
	diag[D_STATE] = 2;
	return 0;
}

int Wrraper_tdc::stop_tdc()
{
	if (stopped) return 0;
	stopped = true;
	printf("stopping the TDC.\n");
	fflush(stdout);
	if (manager) {
		try {
			if (running) manager->Stop();
			manager->ClearBuffer();
			manager->CleanUp(); // cleanup is necessary before delete.
		}
		catch (TDCConfigException& e) { ++diag[D_EXCEPTIONS]; set_error("TDCManager::Stop/CleanUp", e.errorString); }
		catch (...) { ++diag[D_EXCEPTIONS]; set_error("TDCManager::Stop/CleanUp", "unknown exception"); }
		delete manager;
		manager = NULL;
	}
	running = false;
	if (LMF) {
		LMF->CloseOutputLMF();
		delete LMF;
		LMF = NULL;
	}
	diag[D_STATE] = 3;
	return 0;
}


///////////////////////////////////////////////////////////////////
// 25ps GroupMode: the grouping is done inside the TDC driver.
// Same decoding as PCIGetTDC_25psGroupMode_TDC8HP of the RoentDek example.
///////////////////////////////////////////////////////////////////
bool Wrraper_tdc::read_group()
{
	memset(CNT, 0, sizeof(CNT));
	memset(TDC_data, 0, sizeof(TDC_data));

	if (!ui64_HPTDC_AbsoluteTimeStamp) {
		ulRollOver = 0;
		ulAbsoluteTimeStamp = 0;
		ui64_OldTimeStamp = 0;
	}

	// read one group, count = number of words
	++diag[D_READ_CALLS];
	int count = manager->Read(hit_buffer, READ_BUFFER_SIZE);
	if (count < 0) { ++diag[D_READ_NEGATIVE]; return false; }
	if (count == 0) return false;
	++diag[D_READ_NONZERO];
	diag[D_WORDS] += count;
	// keep the last raw words for diagnostics
	n_last_words = (count < N_LAST_WORDS) ? count : N_LAST_WORDS;
	for (int i = 0; i < n_last_words; ++i) last_words[i] = (unsigned __int32)hit_buffer[i];

	bool bOKFlag = false;
	for (int i = 0; i < count; ++i) {
		unsigned __int32 uldata_word = (unsigned __int32)hit_buffer[i];

		if ((uldata_word & 0xC0000000) > 0x40000000) {		// valid data only if rising or falling transition
			++diag[D_WORDS_HIT];
			__int32 lTDCData = (__int32)(uldata_word & 0x00FFFFFF);
			if (lTDCData & 0x00800000) lTDCData |= 0xff000000;	// 24 bit signed -> 32 bit

			unsigned char ucTDCChannel = (unsigned char)((uldata_word & 0x3F000000) >> 24);
			if (ucTDCChannel < NUM_TDC_CHANNELS) {
				++CNT[ucTDCChannel];
				if (CNT[ucTDCChannel] > NUM_TDC_HITS) {
					--CNT[ucTDCChannel];
					++diag[D_WORDS_HIT_OVERFLOW];
				} else {
					TDC_data[ucTDCChannel][CNT[ucTDCChannel] - 1] = lTDCData;
					bOKFlag = true;
				}
			} else {
				++diag[D_WORDS_HIT_BAD_CH];
			}
		} else {
			bool known = false;
			if (uldata_word >> 24 == 0x10) { ulRollOver = uldata_word & 0xffffff; ++diag[D_WORDS_ROLLOVER]; known = true; } // roll-over word
			if (!(uldata_word & 0xf0000000)) { ulAbsoluteTimeStamp = uldata_word & 0xffffff; ++diag[D_WORDS_GROUP]; known = true; } // group word
			if (!known) ++diag[D_WORDS_OTHER];
		}
	}

	unsigned __int64 ui64_TimeStamp_temp = ulRollOver * 0x1000000ull + ulAbsoluteTimeStamp;
	// handle 48 bit roll-overs:
	ui64_HPTDC_AbsoluteTimeStamp += ui64_TimeStamp_temp - ui64_OldTimeStamp;
	if (ui64_TimeStamp_temp < ui64_OldTimeStamp) ui64_HPTDC_AbsoluteTimeStamp += 0x1000000000000ull;
	ui64_OldTimeStamp = ui64_TimeStamp_temp;

	if (bOKFlag) ++diag[D_GROUPS_OK]; else ++diag[D_GROUPS_EMPTY];
	return bOKFlag;
}


int Wrraper_tdc::read_events(unsigned __int64* ts_out, unsigned __int32* cnt_out, __int32* tdc_out, int max_events, int timeout_ms)
{
	if (!running || !manager || stopped) return -1;
	if (max_events <= 0) return 0;
	if (timeout_ms <= 0) timeout_ms = time_out_ms;

	auto t_start = std::chrono::steady_clock::now();
	int n = 0;
	while (n < max_events) {
		bool ok = false;
		try {
			ok = read_group();
		}
		catch (TDCConfigException& e) { ++diag[D_EXCEPTIONS]; set_error("TDCManager::Read", e.errorString); ok = false; }
		catch (...) { ++diag[D_EXCEPTIONS]; set_error("TDCManager::Read", "unknown exception"); ok = false; }
		if (ok) {
			// write the group to the LMF exactly like the old DLL
			if (LMF) {
				LMF->WriteTDCData(ui64_HPTDC_AbsoluteTimeStamp, CNT, &TDC_data[0][0]);
				++diag[D_LMF_EVENTS];
			}
			ts_out[n] = ui64_HPTDC_AbsoluteTimeStamp;
			memcpy(cnt_out + (size_t)n * NUM_TDC_CHANNELS, CNT, sizeof(CNT));
			memcpy(tdc_out + (size_t)n * NUM_TDC_CHANNELS * NUM_TDC_HITS, &TDC_data[0][0], sizeof(TDC_data));
			++n;
		} else {
			Sleep(1);
		}
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start).count();
		if (elapsed >= timeout_ms) break;
	}
	return n;
}


// old interface: [n, then per event ch0..ch7 (first hit, bins), x, y, tof (ns), time stamp (ns)]
double* Wrraper_tdc::get_data_tdc_buf()
{
	int n = read_events(legacy_ts, legacy_cnt, legacy_tdc, buffer_size, time_out_ms);
	if (n < 0) n = 0;
	const double fu = 0.89, fv = 0.917, fw = 0.887, w_offset = -1.175;
	legacy_array[0] = (double)n;
	for (int e = 0; e < n; ++e) {
		const unsigned __int32* c = legacy_cnt + (size_t)e * NUM_TDC_CHANNELS;
		const __int32* t = legacy_tdc + (size_t)e * NUM_TDC_CHANNELS * NUM_TDC_HITS;
		double* row = legacy_array + 1 + (size_t)12 * e;
		for (int ch = 0; ch < NUM_TDC_CHANNELS; ++ch) row[ch] = (double)t[ch * NUM_TDC_HITS];
		double u = 0.5 * fu * (t[0 * NUM_TDC_HITS] - t[1 * NUM_TDC_HITS]) * tdc_binsize;
		double v = 0.5 * fv * (t[2 * NUM_TDC_HITS] - t[3 * NUM_TDC_HITS]) * tdc_binsize;
		double w = 0.5 * fw * ((t[4 * NUM_TDC_HITS] - t[5 * NUM_TDC_HITS]) * tdc_binsize + w_offset);
		row[8] = (2.0 * u + v + w) / 3.0;
		row[9] = (w - v) / sqrt(3.0);
		// TOF = MCP (ch7) - start (ch6); 0 if one of them is missing
		row[10] = (c[6] > 0 && c[7] > 0) ? (t[7 * NUM_TDC_HITS] - t[6 * NUM_TDC_HITS]) * tdc_binsize : 0.0;
		row[11] = (double)legacy_ts[e] * tdc_binsize;
	}
	return legacy_array;
}


// C interface for ctypes
extern "C"
{
	__declspec(dllexport) Wrraper_tdc* Warraper_tdc_new(int buf_size, int time_out) { return new Wrraper_tdc(buf_size, time_out); }
	__declspec(dllexport) void Warraper_tdc_delete(Wrraper_tdc* obj) { delete obj; }
	__declspec(dllexport) int init_tdc(Wrraper_tdc* obj) { return obj->init_tdc(); }
	__declspec(dllexport) int run_tdc(Wrraper_tdc* obj) { return obj->run_tdc(); }
	__declspec(dllexport) int stop_tdc(Wrraper_tdc* obj) { return obj->stop_tdc(); }
	__declspec(dllexport) double* get_data_tdc_buf(Wrraper_tdc* obj) { return obj->get_data_tdc_buf(); }
	__declspec(dllexport) int get_data_tdc_raw(Wrraper_tdc* obj, unsigned __int64* ts, unsigned __int32* cnt, __int32* tdc, int max_events, int timeout_ms)
	{
		return obj->read_events(ts, cnt, tdc, max_events, timeout_ms);
	}
	__declspec(dllexport) double get_bin_size_ns(Wrraper_tdc* obj) { return obj->get_bin_size_ns(); }
	__declspec(dllexport) __int64 get_number_written_events(Wrraper_tdc* obj) { return obj->get_number_written_events(); }
	__declspec(dllexport) int get_wrapper_version() { return WRAPPER_VERSION; }
	__declspec(dllexport) int get_diagnostics(Wrraper_tdc* obj, __int64* out, int n) { return obj->get_diagnostics(out, n); }
	__declspec(dllexport) int get_last_words(Wrraper_tdc* obj, unsigned __int32* out, int n) { return obj->get_last_words(out, n); }
	__declspec(dllexport) const char* get_last_error(Wrraper_tdc* obj) { return obj->get_last_error(); }
	__declspec(dllexport) int get_tdc_info(Wrraper_tdc* obj, double* out, int n) { return obj->get_tdc_info(out, n); }
}
