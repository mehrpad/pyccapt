// Replay an LMF through RoentDek Resort64c and write the sorted hits.
// usage: resort_replay in.lmf params.txt out.bin
#include "LMF_IO.h"
#include "resort64c.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>


std::map<std::string,double> P;
double p(const char* k, double d){ auto it=P.find(k); return it==P.end()? d : it->second; }

void load_table(sort_class* s, const char* fn){
  FILE* f=fopen(fn,"r"); if(!f){printf("no table %s\n",fn);return;}
  char line[512]; int n=0;
  while(fgets(line,sizeof(line),f)){
    char kind[16], layer; double x,y;
    if(sscanf(line,"%15s %c %lf %lf",kind,&layer,&x,&y)!=4) continue;
    interpolate_class* ic=0;
    if(!strcmp(kind,"sum")) ic = layer=='u'? s->signal_corrector->sum_corrector_U : layer=='v'? s->signal_corrector->sum_corrector_V : s->signal_corrector->sum_corrector_W;
    else ic = layer=='u'? s->signal_corrector->pos_corrector_U : layer=='v'? s->signal_corrector->pos_corrector_V : s->signal_corrector->pos_corrector_W;
    ic->set_point(x,y); n++;
  }
  fclose(f); printf("table points %d\n",n);
}

int main(int argc,char** argv){
  if(argc<4){printf("usage\n");return 1;}
  FILE* pf=fopen(argv[2],"r"); char k[128]; double v;
  while(pf && fscanf(pf,"%127s %lf",k,&v)==2) P[k]=v;
  if(pf) fclose(pf);
  LMF_IO* L=new LMF_IO(8,10); if(!L->OpenInputLMF(argv[1])){printf("lmf fail\n");return 1;}
  double bin=L->tdcresolution;
  version_number_class vn=sort_class::get_version_number(); printf("resort64c version %d.%d.%d.%d\n",vn.A,vn.B,vn.C,vn.D);
  const int NI=10; __int32 cnt[8]; double tdc_ns[8][NI];
  sort_class* s=new sort_class();
  s->Cu1=0; s->Cu2=1; s->Cv1=2; s->Cv2=3; s->Cw1=4; s->Cw2=5; s->Cmcp=(int)p("use_mcp",1)? 7 : -1;
  s->use_HEX=true; s->use_MCP=p("use_mcp",1)!=0;
  s->common_start_mode = p("common_start",1)!=0;
  s->MCP_radius=p("mcp_radius",39);
  s->uncorrected_time_sum_half_width_u=p("hw_u",1.5); s->uncorrected_time_sum_half_width_v=p("hw_v",1.2); s->uncorrected_time_sum_half_width_w=p("hw_w",2.3);
  s->corrected_time_sum_half_width_u=p("chw_u",p("hw_u",1.5)); s->corrected_time_sum_half_width_v=p("chw_v",p("hw_v",1.2)); s->corrected_time_sum_half_width_w=p("chw_w",p("hw_w",2.3));
  s->use_sum_correction=p("use_sum_corr",0)!=0; s->use_pos_correction=p("use_pos_corr",0)!=0;
  s->dead_time_anode=p("dead_anode",25); s->dead_time_mcp=p("dead_mcp",25);
  s->fu=p("fu_scale",0.5)*p("fu",0.89); s->fv=p("fu_scale",0.5)*p("fv",0.917); s->fw=p("fu_scale",0.5)*p("fw",0.887);
  s->runtime_u=p("rt_u",88); s->runtime_v=p("rt_v",86); s->runtime_w=p("rt_w",88);
  s->TDC_resolution_ns=p("tdc_res",0.025);
  s->tdc_array_row_length=NI; s->count=cnt; s->tdc_pointer=&tdc_ns[0][0];
  if(p("no_sort",0)!=0) s->run_without_sorting_flag=true;
  int err=s->init_after_setting_parameters();
  if(err){char b[512]; s->get_error_text(err,512,b); printf("init error %d: %s\n",err,b); return 1;}
  if(argc>4) load_table(s,argv[4]);
  double w_off=p("w_offset",-1.175), su=p("sum_u",-113.98), sv=p("sum_v",-113.03), sw=p("sum_w",-114.85), px=p("pos_x",-1.5), py=p("pos_y",1.85);
  FILE* fo=fopen(argv[3],"wb"); long long ev=0; unsigned __int32 c[8]; __int32 d[80];
  srand(12345); int moire=(int)p("anti_moire",0);
  while(L->ReadNextEvent()){
    L->GetNumberOfHitsArray(c); memset(d,0,sizeof(d)); L->GetTDCDataArray(d);
    for(int i=0;i<8;i++){cnt[i]=c[i]; for(int j=0;j<NI;j++){ double r = moire? (rand()/(RAND_MAX+1.0)) : 0.0; tdc_ns[i][j]=(d[i*NI+j]+r)*bin; }}
    double start = cnt[6]>0 ? tdc_ns[6][0] : 0.0;
    s->shift_sums(+1,su,sv,sw); s->shift_layer_w(+1,w_off); s->shift_position_origin(+1,px,py);
    int n = s->use_MCP || 1 ? (p("no_sort",0)!=0 ? s->run_without_sorting() : s->sort()) : 0;
    if(n<0) n=0;
    int nw = n>4?4:n; fwrite(&n,4,1,fo);
    double out[16]; memset(out,0,sizeof(out));
    for(int h=0;h<nw;h++){ hit_class* hh=s->output_hit_array[h]; out[4*h]=hh->x; out[4*h+1]=hh->y; out[4*h+2]=hh->time - start; out[4*h+3]=hh->method; }
    fwrite(out,8,16,fo); ev++;
  }
  fclose(fo); printf("events %lld\n",ev); return 0;
}
