#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TPaveLabel.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TCanvas.h"

#include <iostream>

	
void extractHistos(){
	
	TCanvas * c = new TCanvas();
	c->SetCanvasSize(1200, 600);
	
	TH1::AddDirectory(kFALSE);
	
	TFile * f = TFile::Open("JetNtuple_RunIISummer16_13TeV_MC.root");
	TDirectory * d = f->GetDirectory("AK4jets");
	
	TH1D * matchDR = nullptr;
	TH1D * matchDPT = nullptr;
	TH2D * matchDRDPT = nullptr;
	TH1D * matchDRDist = nullptr;
	TH1D * matchDPTDist = nullptr;
	TH2D * matchDRDPTDist = nullptr;
	TH1D * matchDRPlusDPTDist = nullptr;
	TH2D * matchPTdPTDist = nullptr;
	TH2D * genRecoPT = nullptr;
	TH1D * genDR = nullptr;
	TH1D * genDPhi = nullptr;
	TH1D * genDEta = nullptr;
	TH1D * genPT = nullptr;
	
	
	TH1D * matchPercent = nullptr;
	TH1D * matchNumber = nullptr;
	TH1D * genNumber = nullptr;
	
	d->GetObject("matchDR", matchDR);
	d->GetObject("matchDPT", matchDPT);
	d->GetObject("matchDRDPT", matchDRDPT);
	d->GetObject("matchDRDist", matchDRDist);
	d->GetObject("matchDPTDist", matchDPTDist);
	d->GetObject("matchDRDPTDist", matchDRDPTDist);
	d->GetObject("matchDRPlusDPTDist", matchDRPlusDPTDist);
	d->GetObject("matchPTdPTDist", matchPTdPTDist);
	d->GetObject("genRecoPT", genRecoPT);
	d->GetObject("genDR", genDR);
	d->GetObject("genDPhi", genDPhi);
	d->GetObject("genDEta", genDEta);
	d->GetObject("genPT", genPT);
	
	d->GetObject("matchPercent", matchPercent);
	d->GetObject("matchNumber", matchNumber);
	d->GetObject("genNumber", genNumber);
	
	matchDR->Draw("HIST");
	c->Print("matchDR.png");
	c->Clear();

	matchDPT->Draw("HIST");
	c->Print("matchDPT.png");
	c->Clear();
	
	
	matchDRDPT->Draw("COLZ");
	matchDRDPT->SetStats(0);
	c->Print("matchDRDPT.png");
	c->Clear();
	
	matchDRDPT->SetMaximum(4e4);
	//matchDRDPT->GetXaxis()->SetRangeUser(0,0.8);
	//matchDRDPT->GetYaxis()->SetRangeUser(0,0.8);
	matchDRDPT->Draw("COLZ");
	c->Print("matchDRDPTCutoff.png");
	c->Clear();
	
	
	matchDRDist->Draw("HIST");
	c->Print("matchDRDist.png");
	c->Clear();

	matchDPTDist->Draw("HIST");
	c->Print("matchDPTDist.png");
	c->Clear();
	
	
	matchDRDPTDist->Draw("COLZ");
	matchDRDPTDist->SetStats(0);
	c->Print("matchDRDPTDist.png");
	c->Clear();
	
	matchDRDPTDist->SetMaximum(4e4);
	//matchDRDPTDist->GetXaxis()->SetRangeUser(0,0.8);
	//matchDRDPTDist->GetYaxis()->SetRangeUser(0,0.8);
	matchDRDPTDist->Draw("COLZ");
	c->Print("matchDRDPTDistCutoff.png");
	c->Clear();
	
	matchDRPlusDPTDist->Draw("HIST");
	c->Print("matchDRPlusDPTDist.png");
	c->Clear();
	
	matchPTdPTDist->Draw("COLZ");
	c->Print("matchPTdPTDist.png");
	c->Clear();
	
	genRecoPT->Draw("COLZ");
	c->Print("genRecoPT.png");
	c->Clear();
	
	genDR->Draw("HIST");
	c->Print("genDR.png");
	c->Clear();
	
	genDPhi->Draw("HIST");
	c->Print("genDPhi.png");
	c->Clear();
	
	genDEta->Draw("HIST");
	c->Print("genDEta.png");
	c->Clear();
	
	genPT->Draw("HIST");
	c->Print("genPT.png");
	c->Clear();
	
	
	matchPercent->Draw("HIST");
	c->Print("matchPercent.png");
	c->Clear();
	
	matchNumber->Draw("HIST");
	c->Print("matchNumber.png");
	c->Clear();
	
	genNumber->Draw("HIST");
	c->Print("genNumber.png");
	c->Clear();
	
	
}
