#property copyright "imado"
#property link      ""
#property version   "1.00"

MqlDateTime time;
bool increment = false;
int fileNumber = 0;
int file;

int _ma5;
int _ma15;
int _ma50;
int _ma100;
int _ema5;
int _ema15;
int _ema50;
int _ema100;
int _rsi;
int _stoch;
int _bb;

double ma5[];
double ma15[];
double ma50[];
double ma100[];
double ema5[];
double ema15[];
double ema50[];
double ema100[];
double rsi[];
double stoch1[];
double stoch2[];
double bb1[];
double bb2[];
double bb3[];

MqlRates rates[];

double bid, ask;

int OnInit()
  {
   _ma5 = iMA(Symbol(), Period(), 5, 0, MODE_SMA, PRICE_CLOSE);
   _ma15 = iMA(Symbol(), Period(), 15, 0, MODE_SMA, PRICE_CLOSE);
   _ma50 = iMA(Symbol(), Period(), 50, 0, MODE_SMA, PRICE_CLOSE);
   _ma100 = iMA(Symbol(), Period(), 100, 0, MODE_SMA, PRICE_CLOSE);
   
   _ema5 = iMA(Symbol(), Period(), 5, 0, MODE_EMA, PRICE_CLOSE);
   _ema15 = iMA(Symbol(), Period(), 15, 0, MODE_EMA, PRICE_CLOSE);
   _ema50 = iMA(Symbol(), Period(), 50, 0, MODE_EMA, PRICE_CLOSE);
   _ema100 = iMA(Symbol(), Period(), 100, 0, MODE_EMA, PRICE_CLOSE);
   
   _rsi = iRSI(Symbol(), Period(), 14, PRICE_CLOSE);
   _stoch = iStochastic(Symbol(), Period(), 5, 3, 3, MODE_SMA, STO_LOWHIGH);

   _bb = iBands(Symbol(), Period(), 20, 0, 2.0, PRICE_CLOSE);
   return(INIT_SUCCEEDED);
  }
  
void OnDeinit(const int reason)
  {
   int f = FileOpen("meta.txt", FILE_ANSI|FILE_WRITE|FILE_TXT);
   FileWrite(f, StringFormat("%i", fileNumber - 1));
   FileClose(f);
  }
  
void OnTick()
  {
   TimeGMT(time);
   
   if ((time.hour >= 13 && time.hour <= 17) && (time.day_of_week != 0 && time.day_of_week != 6)){
      if (!increment) {
         increment = true;
         newFile();
      }
      
      writeData();
   } else {
      if (increment) {
         increment = false;
      }
   }
  }
  
void newFile(){
  if (file){
   FileClose(file);
  }
  
   file = FileOpen(StringFormat("%i.csv", fileNumber), FILE_WRITE|FILE_ANSI|FILE_TXT);
   FileWrite(file, "Open,High,Low,Close,TickVolume,MA5,MA15,MA50,MA100,EMA5,EMA15,EMA50,EMA100,RSI,Stoch1,Stoch2,BBU,BBM,BBD,Bid,Ask");
   
   fileNumber ++;
  }
  
void writeData(){
   CopyRates(Symbol(), Period(), 0, 1, rates);
   CopyBuffer(_ma5, 0, 0, 1, ma5);
   CopyBuffer(_ma15, 0, 0, 1, ma15);
   CopyBuffer(_ma50, 0, 0, 1, ma50);
   CopyBuffer(_ma100, 0, 0, 1, ma100);
   CopyBuffer(_ema5, 0, 0, 1, ema5);
   CopyBuffer(_ema15, 0, 0, 1, ema15);
   CopyBuffer(_ema50, 0, 0, 1, ema50);
   CopyBuffer(_ema100, 0, 0, 1, ema100);
   
   CopyBuffer(_rsi, 0, 0, 1, rsi);
   CopyBuffer(_stoch, 0, 0, 1, stoch1);
   CopyBuffer(_stoch, 1, 0, 1, stoch2);

   CopyBuffer(_bb, 1, 0, 1, bb1);
   CopyBuffer(_bb, 0, 0, 1, bb2);
   CopyBuffer(_bb, 2, 0, 1, bb3);

   bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   
   string data = StringFormat("%5f,%5f,%5f,%5f,%li,%5f,%5f,%5f,%5f,%5f,%5f,%5f,%5f,%5f,%5f,%5f,%5f,%5f,%5f,%5f,%5f", rates[0].open, rates[0].high, rates[0].low, rates[0].close, rates[0].tick_volume, ma5[0], ma15[0], ma50[0], ma100[0], ema5[0], ema15[0], ema50[0], ema100[0], rsi[0], stoch1[0], stoch2[0], bb1[0], bb2[0], bb3[0], bid, ask);
   FileWrite(file, data);
}