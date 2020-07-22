

#install.packages('MplusAutomation')
library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(MplusAutomation)



###### Goals ######

# 1. Partition Data into Training and Test Set
# 2. Create Syntax for training dataset
# 3. Run model #2
# 4. Take parameter estimates from #3 and create syntax with those estimates as 
#    fixed values and test dataset
# 5. Run model #4
# 6. Output model fit information from #5 and save to a file
# 7. Repeat 10 times (10-fold CV) 

asmData = read.delim("mydataImpNewCov.dat",header=F,sep="\t")


#variableNames = 'ID comID hhID SA1 IADL1T1 IADL1T2 IADL1T3 IADL1T4 IADL1T5 DEP1T1 DEP1T2 DEP1T3 DEP1T4 DEP1T5 DEP1T6 DEP1T7 DEP1T8 SA2 IADL2T1 IADL2T2 IADL2T3 IADL2T4 IADL2T5 DEP2T1 DEP2T2 DEP2T3 DEP2T4 DEP2T5 DEP2T6 DEP2T7 DEP2T8 SA3 IADL3T1 IADL3T2 IADL3T3 IADL3T4 IADL3T5 DEP3T1 DEP3T2 DEP3T3 DEP3T4 DEP3T5 DEP3T6 DEP3T7 DEP3T8 SEX AGE URBAN MARRIAGE EDU PA1 PA2 PA3'
#variableNames = unlist(strsplit(variableNames, "\\s"))
#dput(variableNames)


names(asmData) = c("ID", "comID", "hhID", "SA1", "IADL1T1", "IADL1T2", "IADL1T3", 
                   "IADL1T4", "IADL1T5", "DEP1T1", "DEP1T2", "DEP1T3", "DEP1T4", 
                   "DEP1T5", "DEP1T6", "DEP1T7", "DEP1T8", "SA2", "IADL2T1", "IADL2T2", 
                   "IADL2T3", "IADL2T4", "IADL2T5", "DEP2T1", "DEP2T2", "DEP2T3", 
                   "DEP2T4", "DEP2T5", "DEP2T6", "DEP2T7", "DEP2T8", "SA3", "IADL3T1", 
                   "IADL3T2", "IADL3T3", "IADL3T4", "IADL3T5", "DEP3T1", "DEP3T2", 
                   "DEP3T3", "DEP3T4", "DEP3T5", "DEP3T6", "DEP3T7", "DEP3T8", "SEX", 
                   "AGE", "URBAN", "MARRIAGE", "EDU", "PA1", "PA2", "PA3")

head(asmData)

# loopReplace function is needed to fill in parameters in Mplus script with specified values
loopReplace <- function(text, replacements) {
	for (v in names(replacements)){
		text <- gsub(sprintf("\\[\\[%s\\]\\]", v), replacements[[v]], text)
	}
	return(text)
}


##################### 10-fold Cross-Validation - Autoregressive Structural Model #########################

## Create Folds
set.seed(20200628)

asmData.2<-asmData[sample(nrow(asmData)),]

folds <- cut(seq(1,nrow(asmData.2)), breaks=10, labels=FALSE)

#Perform 10-fold cross validation

Neg2LL =  matrix(NA,10,1)
CFI  =    matrix(NA,10,1)
SRMR =    matrix(NA,10,1)
RMSEA =   matrix(NA,10,1)
RMSEA_L = matrix(NA,10,1)
RMSEA_U = matrix(NA,10,1)

for(i in 1:10){
    #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- asmData.2[testIndexes, ]
  trainData <- asmData.2[-testIndexes, ]
  
  # Create Syntax for Training Data
  train_script <- mplusObject(
    TITLE = "ASM_plusCovariates;",
    VARIABLE = "USEVARIABLES = SA1 IADL1T1 IADL1T2 IADL1T3 IADL1T4 IADL1T5 DEP1T1 
  DEP1T2 DEP1T3 DEP1T4 DEP1T5 DEP1T6 DEP1T7 DEP1T8 
  SA2 IADL2T1 IADL2T2 IADL2T3 IADL2T4 IADL2T5 
  DEP2T1 DEP2T2 DEP2T3 DEP2T4 DEP2T5 DEP2T6 DEP2T7 DEP2T8 
  SA3 IADL3T1 IADL3T2 IADL3T3 IADL3T4 IADL3T5 
  DEP3T1 DEP3T2 DEP3T3 DEP3T4 DEP3T5 DEP3T6 DEP3T7 DEP3T8 
  SEX AGE URBAN MARRIAGE EDU PA1 PA2 PA3;",
    ANALYSIS = "ESTIMATOR = ML;",
    MODEL = "
    IADL1 BY IADL1T1* (L7)
             IADL1T2  (L8)
             IADL1T3  (L9)
             IADL1T4  (L10)
             IADL1T5  (L11);

    IADL2 BY IADL2T1* (L7)
             IADL2T2  (L8)
             IADL2T3  (L9)
             IADL2T4  (L10)
             IADL2T5  (L11);

    IADL3 BY IADL3T1* (L7)
             IADL3T2  (L8)
             IADL3T3  (L9)
             IADL3T4  (L10)
             IADL3T5  (L11);

    DEP1 BY DEP1T1* (L12)
            DEP1T2  (L13)
            DEP1T3  (L14)
            DEP1T4  (L15)
            DEP1T5  (L16)
            DEP1T6  (L17)
            DEP1T7  (L18)
            DEP1T8  (L19);

    DEP2 BY DEP2T1* (L12)
            DEP2T2  (L13)
            DEP2T3  (L14)
            DEP2T4  (L15)
            DEP2T5  (L16)
            DEP2T6  (L17)
            DEP2T7  (L18)
            DEP2T8  (L19);

    DEP3 BY DEP3T1* (L12)
            DEP3T2  (L13)
            DEP3T3  (L14)
            DEP3T4  (L15)
            DEP3T5  (L16)
            DEP3T6  (L17)
            DEP3T7  (L18)
            DEP3T8  (L19);

    !allow correlated residuals across time for IADL
    IADL1T1 with IADL2T1 IADL3T1; 
    IADL2T1 with IADL3T1;
    IADL1T2 with IADL2T2 IADL3T2; 
    IADL2T2 with IADL3T2;
    IADL1T3 with IADL2T3 IADL3T3; 
    IADL2T3 with IADL3T3;
    IADL1T4 with IADL2T4 IADL3T4; 
    IADL2T4 with IADL3T4;
    IADL1T5 with IADL2T5 IADL3T5; 
    IADL2T5 with IADL3T5;

    !allow correlated residuals across time for DEP
    DEP1T1 with DEP2T1 DEP3T1; 
    DEP2T1 with DEP3T1;
    DEP1T2 with DEP2T2 DEP3T2; 
    DEP2T2 with DEP3T2;
    DEP1T3 with DEP2T3 DEP3T3; 
    DEP2T3 with DEP3T3;
    DEP1T4 with DEP2T4 DEP3T4; 
    DEP2T4 with DEP3T4;
    DEP1T5 with DEP2T5 DEP3T5; 
    DEP2T5 with DEP3T5;
    DEP1T6 with DEP2T6 DEP3T6; 
    DEP2T6 with DEP3T6;
    DEP1T7 with DEP2T7 DEP3T7; 
    DEP2T7 with DEP3T7;
    DEP1T8 with DEP2T8 DEP3T8; 
    DEP2T8 with DEP3T8;

    !modif indices suggest correlated residuals
    IADL1T1 with IADL1T2;
    IADL2T1 with IADL2T2;
    IADL3T1 with IADL3T2;
    DEP1T7 with DEP1T8;
    DEP2T7 with DEP2T8;
    DEP3T7 with DEP3T8;

    !factor variance fixed to 1 for identification
    IADL1@1 ;
    DEP1@1 ;
    
    !latent factor means fixed to 0 for identification
    [IADL1@0 IADL2@0 IADL3@0];
    [DEP1@0 DEP2@0 DEP3@0];

    !intercept constrain across time 
     [IADL1T1-IADL1T5] (I7-I11);
     [IADL2T1-IADL2T5] (K7-K11); 
     [IADL3T1-IADL3T5] (P7-P11);
     [DEP1T1-DEP1T8] (I12-I19);
     [DEP2T1-DEP2T8] (I12-I19); 
     [DEP3T1-DEP3T8] (I12-I19);

     !structural paths (free estiamtion across group and time)
     !corss sectional structural paths 
     !time 1
     IADL1 ON PA1;
     IADL1 ON SA1;
     DEP1 ON PA1;
     DEP1 ON SA1;
     DEP1 ON IADL1;

     !time 2
     IADL2 ON PA2;
     IADL2 ON SA2;
     DEP2 ON PA2;
     DEP2 ON SA2;
     DEP2 ON IADL2;

     !time 3
     IADL3 ON PA3;
     IADL3 ON SA3;
     DEP3 ON PA3;
     DEP3 ON SA3;
     DEP3 ON IADL3;

     !autoregression paths across time AR(2)
     PA2 ON PA1;
     PA3 ON PA2;
     PA3 ON PA1;
     SA2 ON SA1;
     SA3 ON SA2;
     SA3 ON SA1;
     IADL2 ON IADL1;
     IADL3 ON IADL2;
     IADL3 ON IADL1;
     DEP2 ON DEP1;
     DEP3 ON DEP2;
     DEP3 ON DEP1;

     !covariates paths
     PA1 ON SEX AGE URBAN MARRIAGE EDU;
     PA2 ON SEX AGE URBAN MARRIAGE EDU;
     PA3 ON SEX AGE URBAN MARRIAGE EDU;
     SA1 ON SEX AGE URBAN MARRIAGE EDU;
     SA2 ON SEX AGE URBAN MARRIAGE EDU;
     SA3 ON SEX AGE URBAN MARRIAGE EDU;
     IADL1 ON SEX AGE URBAN MARRIAGE EDU;
     IADL2 ON SEX AGE URBAN MARRIAGE EDU;
     IADL3 ON SEX AGE URBAN MARRIAGE EDU;
     DEP1 ON SEX AGE URBAN MARRIAGE EDU;
     DEP2 ON SEX AGE URBAN MARRIAGE EDU;
     DEP3 ON SEX AGE URBAN MARRIAGE EDU;",
    usevariables = c("SA1", "IADL1T1", "IADL1T2", "IADL1T3", 
                     "IADL1T4", "IADL1T5", "DEP1T1", "DEP1T2", "DEP1T3", "DEP1T4", 
                     "DEP1T5", "DEP1T6", "DEP1T7", "DEP1T8", "SA2", "IADL2T1", "IADL2T2", 
                     "IADL2T3", "IADL2T4", "IADL2T5", "DEP2T1", "DEP2T2", "DEP2T3", 
                     "DEP2T4", "DEP2T5", "DEP2T6", "DEP2T7", "DEP2T8", "SA3", "IADL3T1", 
                     "IADL3T2", "IADL3T3", "IADL3T4", "IADL3T5", "DEP3T1", "DEP3T2", 
                     "DEP3T3", "DEP3T4", "DEP3T5", "DEP3T6", "DEP3T7", "DEP3T8", "SEX", 
                     "AGE", "URBAN", "MARRIAGE", "EDU", "PA1", "PA2", "PA3"),
    rdata = trainData)
  
  train.res = mplusModeler(train_script, "trainData.dat", modelout = "trainModel.inp", run = 1L)
  
  parms = train.res$results$parameters	
	
  #create the syntax for loopPeplace to avoid manually input the parameter numbers
  #syntax = data.frame(seq(length(parms$unstandardized$paramHeader)), 
  #                    parms$unstandardized$paramHeader, 
  #                    parms$unstandardized$param, stringsAsFactors = F)
  #names(syntax) = c("rowN", "paramsH", "params")
  #syntax$params <- paste(syntax$paramsH, syntax$params, sep = ' ')
  #syntax$params <- paste(syntax$params,  '@[[', syntax$rowN, ']]', sep = '')
  #syntax$params[syntax$paramsH == 'Intercepts'] <- paste0('[', syntax$params[syntax$paramsH == 'Intercepts'], ']')
  #syntax$params <- paste(syntax$params,  ';\n', sep = '')
  #syntax$params = gsub('Intercepts', '', syntax$params)
  #syntax$params = gsub('Residual.Variances', '', syntax$params)
  #syntax$params = gsub('\\.', ' ', syntax$params)
  #cat(syntax$params)
  
  df_parms <- data.frame(parms[[1]]$param, parms[[1]]$est, 
                         stringsAsFactors = FALSE)
  names(df_parms) <- c("param","est")
  
  df_parms2 <- t(df_parms)
  df_parms2 <- as.data.frame(df_parms2,stringsAsFactors = FALSE)
  df2 <- df_parms2[2,]
  names(df2)<-as.character(seq(length(parms$unstandardized$paramHeader)))
  
  # Create Syntax for Test Data
  
  test_script <- mplusObject(
    TITLE = "ASM on Test Set;",
    VARIABLE = "USEVARIABLES = SA1 IADL1T1 IADL1T2 IADL1T3 IADL1T4 IADL1T5 DEP1T1 
  DEP1T2 DEP1T3 DEP1T4 DEP1T5 DEP1T6 DEP1T7 DEP1T8 
  SA2 IADL2T1 IADL2T2 IADL2T3 IADL2T4 IADL2T5 
  DEP2T1 DEP2T2 DEP2T3 DEP2T4 DEP2T5 DEP2T6 DEP2T7 DEP2T8 
  SA3 IADL3T1 IADL3T2 IADL3T3 IADL3T4 IADL3T5 
  DEP3T1 DEP3T2 DEP3T3 DEP3T4 DEP3T5 DEP3T6 DEP3T7 DEP3T8 
  SEX AGE URBAN MARRIAGE EDU PA1 PA2 PA3;",
    ANALYSIS = "ESTIMATOR = ML;",
    MODEL = loopReplace("
    IADL1 BY IADL1T1@[[1]];
    IADL1 BY IADL1T2@[[2]];
    IADL1 BY IADL1T3@[[3]];
    IADL1 BY IADL1T4@[[4]];
    IADL1 BY IADL1T5@[[5]];
    
    IADL2 BY IADL2T1@[[6]];
    IADL2 BY IADL2T2@[[7]];
    IADL2 BY IADL2T3@[[8]];
    IADL2 BY IADL2T4@[[9]];
    IADL2 BY IADL2T5@[[10]];
    
    IADL3 BY IADL3T1@[[11]];
    IADL3 BY IADL3T2@[[12]];
    IADL3 BY IADL3T3@[[13]];
    IADL3 BY IADL3T4@[[14]];
    IADL3 BY IADL3T5@[[15]];

    DEP1 BY DEP1T1@[[16]];
    DEP1 BY DEP1T2@[[17]];
    DEP1 BY DEP1T3@[[18]];
    DEP1 BY DEP1T4@[[19]];
    DEP1 BY DEP1T5@[[20]];
    DEP1 BY DEP1T6@[[21]];
    DEP1 BY DEP1T7@[[22]];
    DEP1 BY DEP1T8@[[23]];
    
    DEP2 BY DEP2T1@[[24]];
    DEP2 BY DEP2T2@[[25]];
    DEP2 BY DEP2T3@[[26]];
    DEP2 BY DEP2T4@[[27]];
    DEP2 BY DEP2T5@[[28]];
    DEP2 BY DEP2T6@[[29]];
    DEP2 BY DEP2T7@[[30]];
    DEP2 BY DEP2T8@[[31]];
    
    DEP3 BY DEP3T1@[[32]];
    DEP3 BY DEP3T2@[[33]];
    DEP3 BY DEP3T3@[[34]];
    DEP3 BY DEP3T4@[[35]];
    DEP3 BY DEP3T5@[[36]];
    DEP3 BY DEP3T6@[[37]];
    DEP3 BY DEP3T7@[[38]];
    DEP3 BY DEP3T8@[[39]];
    
    DEP1 ON IADL1@[[40]];
    IADL2 ON IADL1@[[41]];
    DEP2 ON IADL2@[[42]];
    DEP2 ON DEP1@[[43]];
    IADL3 ON IADL2@[[44]];
    IADL3 ON IADL1@[[45]];
    DEP3 ON IADL3@[[46]];
    DEP3 ON DEP2@[[47]];
    DEP3 ON DEP1@[[48]];
    
    IADL1 ON PA1@[[49]];
    IADL1 ON SA1@[[50]];
    IADL1 ON SEX@[[51]];
    IADL1 ON AGE@[[52]];
    IADL1 ON URBAN@[[53]];
    IADL1 ON MARRIAGE@[[54]];
    IADL1 ON EDU@[[55]];

    DEP1 ON PA1@[[56]];
    DEP1 ON SA1@[[57]];
    DEP1 ON SEX@[[58]];
    DEP1 ON AGE@[[59]];
    DEP1 ON URBAN@[[60]];
    DEP1 ON MARRIAGE@[[61]];
    DEP1 ON EDU@[[62]];    

    IADL2 ON PA2@[[63]];
    IADL2 ON SA2@[[64]];
    IADL2 ON SEX@[[65]];
    IADL2 ON AGE@[[66]];
    IADL2 ON URBAN@[[67]];
    IADL2 ON MARRIAGE@[[68]];
    IADL2 ON EDU@[[69]];    

    DEP2 ON PA2@[[70]];
    DEP2 ON SA2@[[71]];
    DEP2 ON SEX@[[72]];
    DEP2 ON AGE@[[73]];
    DEP2 ON URBAN@[[74]];
    DEP2 ON MARRIAGE@[[75]];
    DEP2 ON EDU@[[76]];  

    IADL3 ON PA3@[[77]];
    IADL3 ON SA3@[[78]];
    IADL3 ON SEX@[[79]];
    IADL3 ON AGE@[[80]];
    IADL3 ON URBAN@[[81]];
    IADL3 ON MARRIAGE@[[82]];
    IADL3 ON EDU@[[83]];  

    DEP3 ON PA3@[[84]];
    DEP3 ON SA3@[[85]];
    DEP3 ON SEX@[[86]];
    DEP3 ON AGE@[[87]];
    DEP3 ON URBAN@[[88]];
    DEP3 ON MARRIAGE@[[89]];
    DEP3 ON EDU@[[90]];  

    PA2 ON PA1@[[91]];
    PA2 ON SEX@[[92]];
    PA2 ON AGE@[[93]];
    PA2 ON URBAN@[[94]];
    PA2 ON MARRIAGE@[[95]];
    PA2 ON EDU@[[96]];
    
    PA3 ON PA2@[[97]];
    PA3 ON PA1@[[98]];
    PA3 ON SEX@[[99]];
    PA3 ON AGE@[[100]];
    PA3 ON URBAN@[[101]];
    PA3 ON MARRIAGE@[[102]];
    PA3 ON EDU@[[103]];

    SA2 ON SA1@[[104]];
    SA2 ON SEX@[[105]];
    SA2 ON AGE@[[106]];
    SA2 ON URBAN@[[107]];
    SA2 ON MARRIAGE@[[108]];
    SA2 ON EDU@[[109]];

    SA3 ON SA2@[[110]];
    SA3 ON SA1@[[111]];
    SA3 ON SEX@[[112]];
    SA3 ON AGE@[[113]];
    SA3 ON URBAN@[[114]];
    SA3 ON MARRIAGE@[[115]];
    SA3 ON EDU@[[116]];    
    
    PA1 ON SEX@[[117]];
    PA1 ON AGE@[[118]];
    PA1 ON URBAN@[[119]];
    PA1 ON MARRIAGE@[[120]];
    PA1 ON EDU@[[121]];
    
    SA1 ON SEX@[[122]];
    SA1 ON AGE@[[123]];
    SA1 ON URBAN@[[124]];
    SA1 ON MARRIAGE@[[125]];
    SA1 ON EDU@[[126]];  
    
    IADL1T1 WITH IADL2T1@[[127]];  
    IADL1T1 WITH IADL3T1@[[128]];  
    IADL1T1 WITH IADL1T2@[[129]];  
  
    IADL2T1 WITH IADL3T1@[[130]];  
    IADL2T1 WITH IADL2T2@[[131]];  
    
    IADL1T2 WITH IADL3T1@[[132]];  
    IADL1T2 WITH IADL2T2@[[133]];  
    IADL2T2 WITH IADL3T2@[[134]];  
    
    IADL1T3 WITH IADL2T3@[[135]];  
    IADL1T3 WITH IADL3T3@[[136]];  
    IADL2T3 WITH IADL3T3@[[137]];  

    IADL1T4 WITH IADL2T4@[[138]];  
    IADL1T4 WITH IADL3T4@[[139]];      
    IADL2T4 WITH IADL3T4@[[140]];  
  
    IADL1T5 WITH IADL2T5@[[141]];  
    IADL1T5 WITH IADL3T5@[[142]];      
    IADL2T5 WITH IADL3T5@[[143]];   
  
    DEP1T1 WITH DEP2T1@[[144]];
    DEP1T1 WITH DEP3T1@[[145]];
    DEP2T1 WITH DEP3T1@[[146]];
  
    DEP1T2 WITH DEP2T2@[[147]];
    DEP1T2 WITH DEP3T2@[[148]];
    DEP2T2 WITH DEP3T2@[[149]];
    
    DEP1T3 WITH DEP2T3@[[150]];
    DEP1T3 WITH DEP3T3@[[151]];
    DEP2T3 WITH DEP3T3@[[152]];  
    
    DEP1T4 WITH DEP2T4@[[153]];
    DEP1T4 WITH DEP3T4@[[154]];
    DEP2T4 WITH DEP3T4@[[155]];
    
    DEP1T5 WITH DEP2T5@[[156]];
    DEP1T5 WITH DEP3T5@[[157]];
    DEP2T5 WITH DEP3T5@[[158]];
    
    DEP1T6 WITH DEP2T6@[[159]];
    DEP1T6 WITH DEP3T6@[[160]];
    DEP2T6 WITH DEP3T6@[[161]];    
    
    DEP1T7 WITH DEP2T7@[[162]];
    DEP1T7 WITH DEP3T7@[[163]];
    DEP1T7 WITH DEP1T8@[[164]];
    DEP2T7 WITH DEP3T7@[[165]];  
    DEP2T7 WITH DEP2T8@[[166]];
    
    DEP1T8 WITH DEP2T8@[[167]];
    DEP1T8 WITH DEP3T8@[[168]];
    DEP2T8 WITH DEP3T8@[[169]];
    
    IADL3T1 WITH IADL3T2@[[170]];      
    DEP3T7 WITH DEP3T8@[[171]]; 
    
    [SA1@[[172]]];     
    [IADL1T1@[[173]]]; 
    [IADL1T2@[[174]]]; 
    [IADL1T3@[[175]]]; 
    [IADL1T4@[[176]]]; 
    [IADL1T5@[[177]]]; 
    [DEP1T1@[[178]]];  
    [DEP1T2@[[179]]];  
    [DEP1T3@[[180]]];  
    [DEP1T4@[[181]]];  
    [DEP1T5@[[182]]];  
    [DEP1T6@[[183]]];  
    [DEP1T7@[[184]]];  
    [DEP1T8@[[185]]];  
    [SA2@[[186]]];     
    [IADL2T1@[[187]]]; 
    [IADL2T2@[[188]]]; 
    [IADL2T3@[[189]]]; 
    [IADL2T4@[[190]]]; 
    [IADL2T5@[[191]]]; 
    [DEP2T1@[[192]]];  
    [DEP2T2@[[193]]];  
    [DEP2T3@[[194]]];  
    [DEP2T4@[[195]]];  
    [DEP2T5@[[196]]];  
    [DEP2T6@[[197]]];  
    [DEP2T7@[[198]]];  
    [DEP2T8@[[199]]];  
    [SA3@[[200]]];     
    [IADL3T1@[[201]]]; 
    [IADL3T2@[[202]]]; 
    [IADL3T3@[[203]]]; 
    [IADL3T4@[[204]]]; 
    [IADL3T5@[[205]]]; 
    [DEP3T1@[[206]]];  
    [DEP3T2@[[207]]];  
    [DEP3T3@[[208]]];  
    [DEP3T4@[[209]]];  
    [DEP3T5@[[210]]];  
    [DEP3T6@[[211]]];  
    [DEP3T7@[[212]]];  
    [DEP3T8@[[213]]];  
    [PA1@[[214]]];           
    [PA2@[[215]]];           
    [PA3@[[216]]];         
    [IADL1@[[217]]];       
    [IADL2@[[218]]];   
    [IADL3@[[219]]];   
    [DEP1@[[220]]];        
    [DEP2@[[221]]];    
    [DEP3@[[222]]];
    
    SA1@[[223]];     
    IADL1T1@[[224]]; 
    IADL1T2@[[225]]; 
    IADL1T3@[[226]]; 
    IADL1T4@[[227]]; 
    IADL1T5@[[228]]; 
    DEP1T1@[[229]];  
    DEP1T2@[[230]];  
    DEP1T3@[[231]];  
    DEP1T4@[[232]];  
    DEP1T5@[[233]];  
    DEP1T6@[[234]];  
    DEP1T7@[[235]];  
    DEP1T8@[[236]];  
    SA2@[[237]];   
    IADL2T1@[[238]]; 
    IADL2T2@[[239]]; 
    IADL2T3@[[240]]; 
    IADL2T4@[[241]]; 
    IADL2T5@[[242]]; 
    DEP2T1@[[243]];  
    DEP2T2@[[244]];  
    DEP2T3@[[245]];  
    DEP2T4@[[246]];  
    DEP2T5@[[247]];  
    DEP2T6@[[248]];  
    DEP2T7@[[249]];  
    DEP2T8@[[250]];  
    SA3@[[251]];     
    IADL3T1@[[252]]; 
    IADL3T2@[[253]]; 
    IADL3T3@[[254]]; 
    IADL3T4@[[255]]; 
    IADL3T5@[[256]]; 
    DEP3T1@[[257]];  
    DEP3T2@[[258]];  
    DEP3T3@[[259]];  
    DEP3T4@[[260]];  
    DEP3T5@[[261]];  
    DEP3T6@[[262]];  
    DEP3T7@[[263]];  
    DEP3T8@[[264]];  
    PA1@[[265]];           
    PA2@[[266]];           
    PA3@[[267]];         
    IADL1@[[268]];       
    IADL2@[[269]];   
    IADL3@[[270]];   
    DEP1@[[271]];        
    DEP2@[[272]];    
    DEP3@[[273]];    
    OUTPUT: TECH1;", df2),
    
    usevariables = c("SA1", "IADL1T1", "IADL1T2", "IADL1T3", 
                     "IADL1T4", "IADL1T5", "DEP1T1", "DEP1T2", "DEP1T3", "DEP1T4", 
                     "DEP1T5", "DEP1T6", "DEP1T7", "DEP1T8", "SA2", "IADL2T1", "IADL2T2", 
                     "IADL2T3", "IADL2T4", "IADL2T5", "DEP2T1", "DEP2T2", "DEP2T3", 
                     "DEP2T4", "DEP2T5", "DEP2T6", "DEP2T7", "DEP2T8", "SA3", "IADL3T1", 
                     "IADL3T2", "IADL3T3", "IADL3T4", "IADL3T5", "DEP3T1", "DEP3T2", 
                     "DEP3T3", "DEP3T4", "DEP3T5", "DEP3T6", "DEP3T7", "DEP3T8", "SEX", 
                     "AGE", "URBAN", "MARRIAGE", "EDU", "PA1", "PA2", "PA3"),
    rdata = testData)
  
  test.res = mplusModeler(test_script, "testData.dat", modelout = "testModel.inp", run = 1L)
  
  Neg2LL[i,1] = -2 * test.res$results$summaries$LL
  CFI[i,1] = test.res$results$summaries$CFI
  SRMR[i,1] = test.res$results$summaries$SRMR
  RMSEA[i,1] = test.res$results$summaries$RMSEA_Estimate
  RMSEA_L[i,1] = test.res$results$summaries$RMSEA_90CI_LB
  RMSEA_U[i,1] = test.res$results$summaries$RMSEA_90CI_UB
}

warnings()

Neg2LL

mean(Neg2LL)

sd(Neg2LL)

FoldIndex <- matrix(1:10,10,1)

ModelSummary_10F <- data.frame(FoldIndex, CFI, SRMR, RMSEA, RMSEA_L, RMSEA_U)
ModelSummary_10F



############################### plot the results ######################

#Neg2LL_plot<- ggplot(ModelSummary_10F, aes(x=FoldIndex, y=Neg2LL)) +
#                        geom_line() +
#                        scale_x_discrete(limits = factor(1:10)) +
#                        scale_x_continuous(limits = c(0, 10)) +
#                        xlab("Fold") +
#                        ylab("Neg2LL") +
#                        theme(legend.title=element_blank())

CFI_plot<- ggplot(ModelSummary_10F, aes(x=FoldIndex, y=CFI)) +
                        geom_line() +
                        geom_point() +
                        ylim(0.83,0.92) +
                        scale_x_discrete(limits = factor(1:10)) +
                        xlab("Fold") +
                        ylab("CFI") +
                        geom_hline(yintercept=0.9, linetype="dashed", color = "red") + 
                        theme(legend.title=element_blank())

SRMR_plot<- ggplot(ModelSummary_10F, aes(x=FoldIndex, y=SRMR)) +
                       geom_line() +
                       geom_point() +
                       ylim(0.05,0.085) +
                       scale_x_discrete(limits = factor(1:10)) +
                       xlab("Fold")+
                       ylab("SRMR")+
                       geom_hline(yintercept=0.08, linetype="dashed", color = "red") +
                       theme(legend.title=element_blank())

RMSEA_plot<- ggplot(ModelSummary_10F, aes(x=FoldIndex, y=RMSEA)) +
                        geom_line() +
                        geom_point() +
                        ylim(0.025,0.065) +
                        scale_x_discrete(limits = factor(1:10)) +
                        xlab("Fold")+
                        ylab("RMSEA and its 90% confidence interval")+
                        geom_hline(yintercept=0.06, linetype="dashed", color = "red") +
                        theme(legend.title=element_blank())+
                        geom_errorbar(aes(ymin=RMSEA_L, ymax = RMSEA_U), width = .2)


grid.arrange(CFI_plot,  RMSEA_plot, SRMR_plot, nrow = 1)
#ggsave('plots_10Fold.pdf')
                                                                                                                                                                           




                                                                                                                                                                      
  
  
  
  
  
  
  
  
