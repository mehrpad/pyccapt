// Test replacement for the RoentDek TDCManager (only used when MOCK_TDC is defined).
//
// It replays a recorded run instead of talking to the TDC8HP hardware, so the
// complete chain  DLL -> Python -> reconstruction -> HDF5  and the LMF writer
// can be tested on a PC without TDC.
//
// Environment variables:
//   MOCK_TDC_EVENTS       binary event file written by lmf_dump.exe
//                         (per event: uint64 time stamp, uint32 cnt[8], int32 tdc[8][10])
//   MOCK_TDC_MAX_EVENTS   replay at most this many events (default: all)
//   MOCK_TDC_EMPTY_EVERY  every n-th Read() returns 0 words (no data), default 7
//   MOCK_TDC_THROW_INIT   if set, Init() throws a TDCConfigException (tests error reporting)
//   MOCK_TDC_PAUSE_AFTER  after this many events the source is silent for
//                         MOCK_TDC_PAUSE_MS milliseconds (tests the time-out)
//
// Each replayed group is encoded as HPTDC data words exactly like the driver
// delivers them: roll-over word, group word (time stamp), falling-edge hits.

#pragma once
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef unsigned short USHORT;
typedef unsigned long HIT;

class TDCConfigException {
public:
	const char* errorString;
	TDCConfigException(const char* error) { errorString = error; }
};

class TDCInfo {
public:
	int index;
	double resolution;
	int version;
};

class TDCManager {
public:
	TDCManager(USHORT = 0x1A13, USHORT = 0x0001) : pos(0), n_events(0), calls(0), empty_every(7),
		pause_after(-1), pause_ms(0), pause_until(0), paused_done(false) {}

	void Init() {
		if (getenv("MOCK_TDC_THROW_INIT")) throw TDCConfigException("mock: no TDC8HP card found");
		const char* fn = getenv("MOCK_TDC_EVENTS");
		if (fn) {
			FILE* f = fopen(fn, "rb");
			if (f) {
				const size_t rec = 8 + 4 * 8 + 4 * 80;
				std::vector<unsigned char> buf(rec);
				while (fread(buf.data(), 1, rec, f) == rec) data.insert(data.end(), buf.begin(), buf.end());
				fclose(f);
				n_events = (long long)(data.size() / rec);
			}
		}
		const char* m = getenv("MOCK_TDC_MAX_EVENTS");
		if (m && atoll(m) > 0 && atoll(m) < n_events) n_events = atoll(m);
		const char* e = getenv("MOCK_TDC_EMPTY_EVERY");
		if (e) empty_every = atoi(e);
		const char* p = getenv("MOCK_TDC_PAUSE_AFTER");
		if (p) pause_after = atoll(p);
		const char* pm = getenv("MOCK_TDC_PAUSE_MS");
		if (pm) pause_ms = atoi(pm);
		printf("[mock TDC] %lld events loaded\n", n_events);
	}
	TDCInfo getTDCInfo(int) { TDCInfo i; i.index = 0; i.resolution = 0.0251173e-9; i.version = 0x0106; return i; }
	void Reconfigure() {}
	const char* GetParameter(const char* p) {
		if (!_stricmp(p, "GroupRangeEnd")) return "1000";
		if (!_stricmp(p, "GroupRangeStart")) return "-6000";
		if (!_stricmp(p, "Triggerdeadtime")) return "0";
		if (!_stricmp(p, "triggerchannel")) return "7";
		return "0";
	}
	void Start() {}
	void Stop() {}
	void ClearBuffer() {}
	void CleanUp() {}

	int Read(HIT* out, int size) {
		++calls;
		if (empty_every > 0 && (calls % empty_every) == 0) return 0;
		if (pause_after >= 0 && pos == pause_after && !paused_done) {
			if (pause_until == 0) pause_until = GetTickCount64() + pause_ms;
			if (GetTickCount64() < pause_until) return 0;
			paused_done = true;
		}
		if (pos >= n_events) return 0;
		const size_t rec = 8 + 4 * 8 + 4 * 80;
		const unsigned char* r = data.data() + rec * (size_t)pos;
		unsigned long long ts; memcpy(&ts, r, 8);
		unsigned int cnt[8]; memcpy(cnt, r + 8, 32);
		int tdc[80]; memcpy(tdc, r + 40, 320);
		int n = 0;
		out[n++] = (HIT)(0x10000000u | (unsigned int)((ts >> 24) & 0xffffff));  // roll-over word
		out[n++] = (HIT)(unsigned int)(ts & 0xffffff);                          // group word
		for (int ch = 0; ch < 8; ++ch)
			for (unsigned int h = 0; h < cnt[ch] && h < 10; ++h)
				if (n < size) out[n++] = (HIT)(0x80000000u | ((unsigned int)ch << 24) | ((unsigned int)tdc[ch * 10 + h] & 0x00ffffff));
		++pos;
		return n;
	}

private:
	std::vector<unsigned char> data;
	long long pos, n_events, calls;
	int empty_every;
	long long pause_after;
	int pause_ms;
	unsigned long long pause_until;
	bool paused_done;
};
