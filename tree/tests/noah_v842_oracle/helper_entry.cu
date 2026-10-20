extern "C" __global__ void helper_noah_fac2mit(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*2;int* ix=ixs+c*0;NoahTables tables=*ts;tables.error=0;
float SMCMAX_arg=x[0];float FLIMIT_arg=x[1];
noah_fac2mit(tables,SMCMAX_arg,FLIMIT_arg);
x[0]=SMCMAX_arg;x[1]=FLIMIT_arg;
status[c]=tables.error;
}

extern "C" __global__ void helper_noah_frh2o(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*7;int* ix=ixs+c*0;NoahTables tables=*ts;tables.error=0;
float FREE_arg=x[0];float TKELV_arg=x[1];float SMC_arg=x[2];float SH2O_arg=x[3];float SMCMAX_arg=x[4];float BEXP_arg=x[5];float PSIS_arg=x[6];
noah_frh2o(tables,FREE_arg,TKELV_arg,SMC_arg,SH2O_arg,SMCMAX_arg,BEXP_arg,PSIS_arg);
x[0]=FREE_arg;x[1]=TKELV_arg;x[2]=SMC_arg;x[3]=SH2O_arg;x[4]=SMCMAX_arg;x[5]=BEXP_arg;x[6]=PSIS_arg;
status[c]=tables.error;
}

extern "C" __global__ void helper_gl_snow_new(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*4;int* ix=ixs+c*0;NoahTables tables=*ts;tables.error=0;
float TEMP_arg=x[0];float NEWSN_arg=x[1];float SNOWH_arg=x[2];float SNDENS_arg=x[3];
gl_snow_new(tables,TEMP_arg,NEWSN_arg,SNOWH_arg,SNDENS_arg);
x[0]=TEMP_arg;x[1]=NEWSN_arg;x[2]=SNOWH_arg;x[3]=SNDENS_arg;
status[c]=tables.error;
}

extern "C" __global__ void helper_noah_evapo(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*43;int* ix=ixs+c*2;NoahTables tables=*ts;tables.error=0;
float ETA1_arg=x[0];int NSOIL_arg=ix[0];float CMC_arg=x[5];float ETP1_arg=x[6];float DT_arg=x[7];float SMCMAX_arg=x[16];float BEXP_arg=x[17];float PC_arg=x[18];float SMCWLT_arg=x[19];float DKSAT_arg=x[20];float DWSAT_arg=x[21];float SMCREF_arg=x[22];float SHDFAC_arg=x[23];float CMCMAX_arg=x[24];float SMCDRY_arg=x[25];float CFACTR_arg=x[26];float EDIR_arg=x[27];float EC_arg=x[28];float ETT_arg=x[33];float SFCTMP_arg=x[34];float Q2_arg=x[35];int NROOT_arg=ix[1];float FXEXP_arg=x[40];float SFHEAD1RT_arg=x[41];float ETPND1_arg=x[42];
noah_evapo(tables,ETA1_arg,x+1,NSOIL_arg,CMC_arg,ETP1_arg,DT_arg,x+8,x+12,SMCMAX_arg,BEXP_arg,PC_arg,SMCWLT_arg,DKSAT_arg,DWSAT_arg,SMCREF_arg,SHDFAC_arg,CMCMAX_arg,SMCDRY_arg,CFACTR_arg,EDIR_arg,EC_arg,x+29,ETT_arg,SFCTMP_arg,Q2_arg,NROOT_arg,x+36,FXEXP_arg,SFHEAD1RT_arg,ETPND1_arg);
x[0]=ETA1_arg;ix[0]=NSOIL_arg;x[5]=CMC_arg;x[6]=ETP1_arg;x[7]=DT_arg;x[16]=SMCMAX_arg;x[17]=BEXP_arg;x[18]=PC_arg;x[19]=SMCWLT_arg;x[20]=DKSAT_arg;x[21]=DWSAT_arg;x[22]=SMCREF_arg;x[23]=SHDFAC_arg;x[24]=CMCMAX_arg;x[25]=SMCDRY_arg;x[26]=CFACTR_arg;x[27]=EDIR_arg;x[28]=EC_arg;x[33]=ETT_arg;x[34]=SFCTMP_arg;x[35]=Q2_arg;ix[1]=NROOT_arg;x[40]=FXEXP_arg;x[41]=SFHEAD1RT_arg;x[42]=ETPND1_arg;
status[c]=tables.error;
}

extern "C" __global__ void helper_noah_nopac(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*81;int* ix=ixs+c*7;NoahTables tables=*ts;tables.error=0;
float ETP_arg=x[0];float ETA_arg=x[1];float PRCP_arg=x[2];float SMCMAX_arg=x[7];float SMCWLT_arg=x[8];float SMCREF_arg=x[9];float SMCDRY_arg=x[10];float CMC_arg=x[11];float CMCMAX_arg=x[12];int NSOIL_arg=ix[0];float DT_arg=x[13];float SHDFAC_arg=x[14];float SBETA_arg=x[15];float Q2_arg=x[16];float T1_arg=x[17];float SFCTMP_arg=x[18];float T24_arg=x[19];float TH2_arg=x[20];float FDOWN_arg=x[21];float F1_arg=x[22];float EMISSI_arg=x[23];float SSOIL_arg=x[24];float EPSCA_arg=x[29];float BEXP_arg=x[30];float PC_arg=x[31];float RCH_arg=x[32];float RR_arg=x[33];float CFACTR_arg=x[34];float SLOPE_arg=x[39];float KDT_arg=x[40];float FRZFACT_arg=x[41];float PSISAT_arg=x[42];float DKSAT_arg=x[47];float DWSAT_arg=x[48];float TBOT_arg=x[49];float ZBOT_arg=x[50];float RUNOFF1_arg=x[51];float RUNOFF2_arg=x[52];float RUNOFF3_arg=x[53];float EDIR_arg=x[54];float EC_arg=x[55];float ETT_arg=x[60];int NROOT_arg=ix[1];float QUARTZ_arg=x[65];float FXEXP_arg=x[66];float CSOIL_arg=x[67];float BETA_arg=x[68];float DRIP_arg=x[69];float DEW_arg=x[70];float FLX1_arg=x[71];float FLX3_arg=x[72];int VEGTYP_arg=ix[2];int ISURBAN_arg=ix[3];float SFHEAD1RT_arg=x[73];float INFXS1RT_arg=x[74];float ETPND1_arg=x[75];int SOILTYP_arg=ix[4];int OPT_THCND_arg=ix[5];float XSDA_QFX_arg=x[76];float QFX_PHY_arg=x[77];float XQNORM_arg=x[78];int FASDAS_arg=ix[6];float HCPCT_FASDAS_arg=x[79];float IRRIGATION_CHANNEL_arg=x[80];
noah_nopac(tables,ETP_arg,ETA_arg,PRCP_arg,x+3,SMCMAX_arg,SMCWLT_arg,SMCREF_arg,SMCDRY_arg,CMC_arg,CMCMAX_arg,NSOIL_arg,DT_arg,SHDFAC_arg,SBETA_arg,Q2_arg,T1_arg,SFCTMP_arg,T24_arg,TH2_arg,FDOWN_arg,F1_arg,EMISSI_arg,SSOIL_arg,x+25,EPSCA_arg,BEXP_arg,PC_arg,RCH_arg,RR_arg,CFACTR_arg,x+35,SLOPE_arg,KDT_arg,FRZFACT_arg,PSISAT_arg,x+43,DKSAT_arg,DWSAT_arg,TBOT_arg,ZBOT_arg,RUNOFF1_arg,RUNOFF2_arg,RUNOFF3_arg,EDIR_arg,EC_arg,x+56,ETT_arg,NROOT_arg,x+61,QUARTZ_arg,FXEXP_arg,CSOIL_arg,BETA_arg,DRIP_arg,DEW_arg,FLX1_arg,FLX3_arg,VEGTYP_arg,ISURBAN_arg,SFHEAD1RT_arg,INFXS1RT_arg,ETPND1_arg,SOILTYP_arg,OPT_THCND_arg,XSDA_QFX_arg,QFX_PHY_arg,XQNORM_arg,FASDAS_arg,HCPCT_FASDAS_arg,IRRIGATION_CHANNEL_arg,true);
x[0]=ETP_arg;x[1]=ETA_arg;x[2]=PRCP_arg;x[7]=SMCMAX_arg;x[8]=SMCWLT_arg;x[9]=SMCREF_arg;x[10]=SMCDRY_arg;x[11]=CMC_arg;x[12]=CMCMAX_arg;ix[0]=NSOIL_arg;x[13]=DT_arg;x[14]=SHDFAC_arg;x[15]=SBETA_arg;x[16]=Q2_arg;x[17]=T1_arg;x[18]=SFCTMP_arg;x[19]=T24_arg;x[20]=TH2_arg;x[21]=FDOWN_arg;x[22]=F1_arg;x[23]=EMISSI_arg;x[24]=SSOIL_arg;x[29]=EPSCA_arg;x[30]=BEXP_arg;x[31]=PC_arg;x[32]=RCH_arg;x[33]=RR_arg;x[34]=CFACTR_arg;x[39]=SLOPE_arg;x[40]=KDT_arg;x[41]=FRZFACT_arg;x[42]=PSISAT_arg;x[47]=DKSAT_arg;x[48]=DWSAT_arg;x[49]=TBOT_arg;x[50]=ZBOT_arg;x[51]=RUNOFF1_arg;x[52]=RUNOFF2_arg;x[53]=RUNOFF3_arg;x[54]=EDIR_arg;x[55]=EC_arg;x[60]=ETT_arg;ix[1]=NROOT_arg;x[65]=QUARTZ_arg;x[66]=FXEXP_arg;x[67]=CSOIL_arg;x[68]=BETA_arg;x[69]=DRIP_arg;x[70]=DEW_arg;x[71]=FLX1_arg;x[72]=FLX3_arg;ix[2]=VEGTYP_arg;ix[3]=ISURBAN_arg;x[73]=SFHEAD1RT_arg;x[74]=INFXS1RT_arg;x[75]=ETPND1_arg;ix[4]=SOILTYP_arg;ix[5]=OPT_THCND_arg;x[76]=XSDA_QFX_arg;x[77]=QFX_PHY_arg;x[78]=XQNORM_arg;ix[6]=FASDAS_arg;x[79]=HCPCT_FASDAS_arg;x[80]=IRRIGATION_CHANNEL_arg;
status[c]=tables.error;
}

extern "C" __global__ void helper_noah_snopac(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*93;int* ix=ixs+c*9;NoahTables tables=*ts;tables.error=0;
float ETP_arg=x[0];float ETA_arg=x[1];float PRCP_arg=x[2];float PRCPF_arg=x[3];bool SNOWNG_arg=ix[0];float SMCMAX_arg=x[8];float SMCWLT_arg=x[9];float SMCREF_arg=x[10];float SMCDRY_arg=x[11];float CMC_arg=x[12];float CMCMAX_arg=x[13];int NSOIL_arg=ix[1];float DT_arg=x[14];float SBETA_arg=x[15];float DF1_arg=x[16];float Q2_arg=x[17];float T1_arg=x[18];float SFCTMP_arg=x[19];float T24_arg=x[20];float TH2_arg=x[21];float FDOWN_arg=x[22];float F1_arg=x[23];float SSOIL_arg=x[24];float EPSCA_arg=x[29];float SFCPRS_arg=x[30];float BEXP_arg=x[31];float PC_arg=x[32];float RCH_arg=x[33];float RR_arg=x[34];float CFACTR_arg=x[35];float SNCOVR_arg=x[36];float ESD_arg=x[37];float SNDENS_arg=x[38];float SNOWH_arg=x[39];float SLOPE_arg=x[44];float KDT_arg=x[45];float FRZFACT_arg=x[46];float PSISAT_arg=x[47];float DWSAT_arg=x[52];float DKSAT_arg=x[53];float TBOT_arg=x[54];float ZBOT_arg=x[55];float SHDFAC_arg=x[56];float RUNOFF1_arg=x[57];float RUNOFF2_arg=x[58];float RUNOFF3_arg=x[59];float EDIR_arg=x[60];float EC_arg=x[61];float ETT_arg=x[66];int NROOT_arg=ix[2];float SNOMLT_arg=x[67];float QUARTZ_arg=x[72];float FXEXP_arg=x[73];float CSOIL_arg=x[74];float BETA_arg=x[75];float DRIP_arg=x[76];float DEW_arg=x[77];float FLX1_arg=x[78];float FLX2_arg=x[79];float FLX3_arg=x[80];float ESNOW_arg=x[81];float ETNS_arg=x[82];float EMISSI_arg=x[83];float RIBB_arg=x[84];float SOLDN_arg=x[85];int ISURBAN_arg=ix[3];int VEGTYP_arg=ix[4];float ETPN_arg=x[86];float FLX4_arg=x[87];bool UA_PHYS_arg=ix[5];float SFHEAD1RT_arg=x[88];float INFXS1RT_arg=x[89];float ETPND1_arg=x[90];int SOILTYP_arg=ix[6];int OPT_THCND_arg=ix[7];float QFX_PHY_arg=x[91];int FASDAS_arg=ix[8];float HCPCT_FASDAS_arg=x[92];
noah_snopac(tables,ETP_arg,ETA_arg,PRCP_arg,PRCPF_arg,SNOWNG_arg,x+4,SMCMAX_arg,SMCWLT_arg,SMCREF_arg,SMCDRY_arg,CMC_arg,CMCMAX_arg,NSOIL_arg,DT_arg,SBETA_arg,DF1_arg,Q2_arg,T1_arg,SFCTMP_arg,T24_arg,TH2_arg,FDOWN_arg,F1_arg,SSOIL_arg,x+25,EPSCA_arg,SFCPRS_arg,BEXP_arg,PC_arg,RCH_arg,RR_arg,CFACTR_arg,SNCOVR_arg,ESD_arg,SNDENS_arg,SNOWH_arg,x+40,SLOPE_arg,KDT_arg,FRZFACT_arg,PSISAT_arg,x+48,DWSAT_arg,DKSAT_arg,TBOT_arg,ZBOT_arg,SHDFAC_arg,RUNOFF1_arg,RUNOFF2_arg,RUNOFF3_arg,EDIR_arg,EC_arg,x+62,ETT_arg,NROOT_arg,SNOMLT_arg,x+68,QUARTZ_arg,FXEXP_arg,CSOIL_arg,BETA_arg,DRIP_arg,DEW_arg,FLX1_arg,FLX2_arg,FLX3_arg,ESNOW_arg,ETNS_arg,EMISSI_arg,RIBB_arg,SOLDN_arg,ISURBAN_arg,VEGTYP_arg,ETPN_arg,FLX4_arg,UA_PHYS_arg,SFHEAD1RT_arg,INFXS1RT_arg,ETPND1_arg,SOILTYP_arg,OPT_THCND_arg,QFX_PHY_arg,FASDAS_arg,HCPCT_FASDAS_arg);
x[0]=ETP_arg;x[1]=ETA_arg;x[2]=PRCP_arg;x[3]=PRCPF_arg;ix[0]=SNOWNG_arg;x[8]=SMCMAX_arg;x[9]=SMCWLT_arg;x[10]=SMCREF_arg;x[11]=SMCDRY_arg;x[12]=CMC_arg;x[13]=CMCMAX_arg;ix[1]=NSOIL_arg;x[14]=DT_arg;x[15]=SBETA_arg;x[16]=DF1_arg;x[17]=Q2_arg;x[18]=T1_arg;x[19]=SFCTMP_arg;x[20]=T24_arg;x[21]=TH2_arg;x[22]=FDOWN_arg;x[23]=F1_arg;x[24]=SSOIL_arg;x[29]=EPSCA_arg;x[30]=SFCPRS_arg;x[31]=BEXP_arg;x[32]=PC_arg;x[33]=RCH_arg;x[34]=RR_arg;x[35]=CFACTR_arg;x[36]=SNCOVR_arg;x[37]=ESD_arg;x[38]=SNDENS_arg;x[39]=SNOWH_arg;x[44]=SLOPE_arg;x[45]=KDT_arg;x[46]=FRZFACT_arg;x[47]=PSISAT_arg;x[52]=DWSAT_arg;x[53]=DKSAT_arg;x[54]=TBOT_arg;x[55]=ZBOT_arg;x[56]=SHDFAC_arg;x[57]=RUNOFF1_arg;x[58]=RUNOFF2_arg;x[59]=RUNOFF3_arg;x[60]=EDIR_arg;x[61]=EC_arg;x[66]=ETT_arg;ix[2]=NROOT_arg;x[67]=SNOMLT_arg;x[72]=QUARTZ_arg;x[73]=FXEXP_arg;x[74]=CSOIL_arg;x[75]=BETA_arg;x[76]=DRIP_arg;x[77]=DEW_arg;x[78]=FLX1_arg;x[79]=FLX2_arg;x[80]=FLX3_arg;x[81]=ESNOW_arg;x[82]=ETNS_arg;x[83]=EMISSI_arg;x[84]=RIBB_arg;x[85]=SOLDN_arg;ix[3]=ISURBAN_arg;ix[4]=VEGTYP_arg;x[86]=ETPN_arg;x[87]=FLX4_arg;ix[5]=UA_PHYS_arg;x[88]=SFHEAD1RT_arg;x[89]=INFXS1RT_arg;x[90]=ETPND1_arg;ix[6]=SOILTYP_arg;ix[7]=OPT_THCND_arg;x[91]=QFX_PHY_arg;ix[8]=FASDAS_arg;x[92]=HCPCT_FASDAS_arg;
status[c]=tables.error;
}

extern "C" __global__ void helper_noah_penman(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*29;int* ix=ixs+c*3;NoahTables tables=*ts;tables.error=0;
float SFCTMP_arg=x[0];float SFCPRS_arg=x[1];float CH_arg=x[2];float T2V_arg=x[3];float TH2_arg=x[4];float PRCP_arg=x[5];float FDOWN_arg=x[6];float T24_arg=x[7];float SSOIL_arg=x[8];float Q2_arg=x[9];float Q2SAT_arg=x[10];float ETP_arg=x[11];float RCH_arg=x[12];float EPSCA_arg=x[13];float RR_arg=x[14];bool SNOWNG_arg=ix[0];bool FRZGRA_arg=ix[1];float DQSDT2_arg=x[15];float FLX2_arg=x[16];float EMISSI_IN_arg=x[17];float SNEQV_arg=x[18];float T1_arg=x[19];float SNCOVR_arg=x[20];float AOASIS_arg=x[21];float ALBEDO_arg=x[22];float SOLDN_arg=x[23];float FVB_arg=x[24];float GAMA_arg=x[25];float STC1_arg=x[26];float ETPN_arg=x[27];float FLX4_arg=x[28];bool UA_PHYS_arg=ix[2];
noah_penman(tables,SFCTMP_arg,SFCPRS_arg,CH_arg,T2V_arg,TH2_arg,PRCP_arg,FDOWN_arg,T24_arg,SSOIL_arg,Q2_arg,Q2SAT_arg,ETP_arg,RCH_arg,EPSCA_arg,RR_arg,SNOWNG_arg,FRZGRA_arg,DQSDT2_arg,FLX2_arg,EMISSI_IN_arg,SNEQV_arg,T1_arg,SNCOVR_arg,AOASIS_arg,ALBEDO_arg,SOLDN_arg,FVB_arg,GAMA_arg,STC1_arg,ETPN_arg,FLX4_arg,UA_PHYS_arg);
x[0]=SFCTMP_arg;x[1]=SFCPRS_arg;x[2]=CH_arg;x[3]=T2V_arg;x[4]=TH2_arg;x[5]=PRCP_arg;x[6]=FDOWN_arg;x[7]=T24_arg;x[8]=SSOIL_arg;x[9]=Q2_arg;x[10]=Q2SAT_arg;x[11]=ETP_arg;x[12]=RCH_arg;x[13]=EPSCA_arg;x[14]=RR_arg;ix[0]=SNOWNG_arg;ix[1]=FRZGRA_arg;x[15]=DQSDT2_arg;x[16]=FLX2_arg;x[17]=EMISSI_IN_arg;x[18]=SNEQV_arg;x[19]=T1_arg;x[20]=SNCOVR_arg;x[21]=AOASIS_arg;x[22]=ALBEDO_arg;x[23]=SOLDN_arg;x[24]=FVB_arg;x[25]=GAMA_arg;x[26]=STC1_arg;x[27]=ETPN_arg;x[28]=FLX4_arg;ix[2]=UA_PHYS_arg;
status[c]=tables.error;
}

extern "C" __global__ void helper_noah_tdfcnd(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*7;int* ix=ixs+c*2;NoahTables tables=*ts;tables.error=0;
float DF_arg=x[0];float SMC_arg=x[1];float QZ_arg=x[2];float SMCMAX_arg=x[3];float SH2O_arg=x[4];float BEXP_arg=x[5];float PSISAT_arg=x[6];int SOILTYP_arg=ix[0];int OPT_THCND_arg=ix[1];
noah_tdfcnd(tables,DF_arg,SMC_arg,QZ_arg,SMCMAX_arg,SH2O_arg,BEXP_arg,PSISAT_arg,SOILTYP_arg,OPT_THCND_arg);
x[0]=DF_arg;x[1]=SMC_arg;x[2]=QZ_arg;x[3]=SMCMAX_arg;x[4]=SH2O_arg;x[5]=BEXP_arg;x[6]=PSISAT_arg;ix[0]=SOILTYP_arg;ix[1]=OPT_THCND_arg;
status[c]=tables.error;
}

extern "C" __global__ void helper_noah_alcalc(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*12;int* ix=ixs+c*1;NoahTables tables=*ts;tables.error=0;
float ALB_arg=x[0];float SNOALB_arg=x[1];float EMBRD_arg=x[2];float SHDFAC_arg=x[3];float SHDMIN_arg=x[4];float SNCOVR_arg=x[5];float TSNOW_arg=x[6];float ALBEDO_arg=x[7];float EMISSI_arg=x[8];float DT_arg=x[9];bool SNOWNG_arg=ix[0];float SNOTIME1_arg=x[10];float LVCOEF_arg=x[11];
noah_alcalc(tables,ALB_arg,SNOALB_arg,EMBRD_arg,SHDFAC_arg,SHDMIN_arg,SNCOVR_arg,TSNOW_arg,ALBEDO_arg,EMISSI_arg,DT_arg,SNOWNG_arg,SNOTIME1_arg,LVCOEF_arg);
x[0]=ALB_arg;x[1]=SNOALB_arg;x[2]=EMBRD_arg;x[3]=SHDFAC_arg;x[4]=SHDMIN_arg;x[5]=SNCOVR_arg;x[6]=TSNOW_arg;x[7]=ALBEDO_arg;x[8]=EMISSI_arg;x[9]=DT_arg;ix[0]=SNOWNG_arg;x[10]=SNOTIME1_arg;x[11]=LVCOEF_arg;
status[c]=tables.error;
}

extern "C" __global__ void helper_gl_alcalc(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*8;int* ix=ixs+c*1;NoahTables tables=*ts;tables.error=0;
float ALB_arg=x[0];float SNOALB_arg=x[1];float EMBRD_arg=x[2];float TSNOW_arg=x[3];float ALBEDO_arg=x[4];float EMISSI_arg=x[5];float DT_arg=x[6];bool SNOWNG_arg=ix[0];float SNOTIME1_arg=x[7];
gl_alcalc(tables,ALB_arg,SNOALB_arg,EMBRD_arg,TSNOW_arg,ALBEDO_arg,EMISSI_arg,DT_arg,SNOWNG_arg,SNOTIME1_arg);
x[0]=ALB_arg;x[1]=SNOALB_arg;x[2]=EMBRD_arg;x[3]=TSNOW_arg;x[4]=ALBEDO_arg;x[5]=EMISSI_arg;x[6]=DT_arg;ix[0]=SNOWNG_arg;x[7]=SNOTIME1_arg;
status[c]=tables.error;
}

extern "C" __global__ void helper_noah_devap_hydro(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*16;int* ix=ixs+c*0;NoahTables tables=*ts;tables.error=0;
float EDIR_arg=x[0];float ETP1_arg=x[1];float SMC_arg=x[2];float ZSOIL_arg=x[3];float SHDFAC_arg=x[4];float SMCMAX_arg=x[5];float BEXP_arg=x[6];float DKSAT_arg=x[7];float DWSAT_arg=x[8];float SMCDRY_arg=x[9];float SMCREF_arg=x[10];float SMCWLT_arg=x[11];float FXEXP_arg=x[12];float SFHEAD1RT_arg=x[13];float ETPND1_arg=x[14];float DT_arg=x[15];
noah_devap_hydro(tables,EDIR_arg,ETP1_arg,SMC_arg,ZSOIL_arg,SHDFAC_arg,SMCMAX_arg,BEXP_arg,DKSAT_arg,DWSAT_arg,SMCDRY_arg,SMCREF_arg,SMCWLT_arg,FXEXP_arg,SFHEAD1RT_arg,ETPND1_arg,DT_arg);
x[0]=EDIR_arg;x[1]=ETP1_arg;x[2]=SMC_arg;x[3]=ZSOIL_arg;x[4]=SHDFAC_arg;x[5]=SMCMAX_arg;x[6]=BEXP_arg;x[7]=DKSAT_arg;x[8]=DWSAT_arg;x[9]=SMCDRY_arg;x[10]=SMCREF_arg;x[11]=SMCWLT_arg;x[12]=FXEXP_arg;x[13]=SFHEAD1RT_arg;x[14]=ETPND1_arg;x[15]=DT_arg;
status[c]=tables.error;
}

extern "C" __global__ void helper_noah_snow_new(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*4;int* ix=ixs+c*0;NoahTables tables=*ts;tables.error=0;
float TEMP_arg=x[0];float NEWSN_arg=x[1];float SNOWH_arg=x[2];float SNDENS_arg=x[3];
noah_snow_new(tables,TEMP_arg,NEWSN_arg,SNOWH_arg,SNDENS_arg);
x[0]=TEMP_arg;x[1]=NEWSN_arg;x[2]=SNOWH_arg;x[3]=SNDENS_arg;
status[c]=tables.error;
}
