// Dump an LMF (TDC8HP, 8ch x 10 hits) to a flat binary file:
// per event: uint64 timestamp, uint32 cnt[8], int32 tdc[8][10]
#include "LMF_IO.h"
#include <cstdio>
#include <cstring>
int main(int argc, char** argv){
  if(argc<3){printf("usage: lmf_dump in.lmf out.bin\n");return 1;}
  LMF_IO* R=new LMF_IO(8,10);
  if(!R->OpenInputLMF(argv[1])){printf("open fail\n");return 1;}
  printf("channels %u hits %u res %g events %lld\n",R->GetNumberOfChannels(),R->GetMaxNumberOfHits(),R->tdcresolution,(long long)R->uint64_Numberofevents);
  FILE* f=fopen(argv[2],"wb"); long long n=0;
  unsigned __int32 c[8]; __int32 d[80];
  while(R->ReadNextEvent()){
    unsigned __int64 ts=R->Getuint64TimeStamp();
    R->GetNumberOfHitsArray(c); memset(d,0,sizeof(d)); R->GetTDCDataArray(d);
    fwrite(&ts,8,1,f); fwrite(c,4,8,f); fwrite(d,4,80,f); n++;
  }
  fclose(f); printf("written %lld\n",n); return 0;
}
