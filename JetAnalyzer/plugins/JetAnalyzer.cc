//  Jet tuple producer for 13 TeV Run2 MC samples
//  Specifically aimed at studies of gluon and light quark jets
//  Data is saved to file on a jet-by-jet basis, resulting in almost flat tuples
//
//  Author: Kimmo Kallonen
//  Based on previous work by: Petra-Maria Ekroos

#include "JetAnalyzer.h"
//#include "TH1D.h"
//#include "TH2D.h"

JetAnalyzer::JetAnalyzer(const edm::ParameterSet& iConfig):
    vtxToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
    jetToken_(consumes<pat::JetCollection>(iConfig.getParameter<edm::InputTag>("jets"))),
    pfToken_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCands"))),
    EDMGenJetsToken_(consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("genJets"))),
    genEventInfoToken_(consumes <GenEventInfoProduct>(iConfig.getParameter<edm::InputTag>("genEventInfo"))),
    pileupInfoToken_(consumes <std::vector<PileupSummaryInfo>> (iConfig.getParameter<edm::InputTag>("pileupInfo"))),
    pfRhoAllToken_(consumes <double> (iConfig.getParameter<edm::InputTag>("pfRhoAll"))),
    pfRhoCentralToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("pfRhoCentral"))),
    pfRhoCentralNeutralToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("pfRhoCentralNeutral"))),
    pfRhoCentralChargedPileUpToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("pfRhoCentralChargedPileUp"))),
    qglToken_(consumes<edm::ValueMap<float>>(edm::InputTag("QGTagger", "qgLikelihood"))),
    ptDToken_(consumes<edm::ValueMap<float>>(edm::InputTag("QGTagger", "ptD"))),
    axis2Token_(consumes<edm::ValueMap<float>>(edm::InputTag("QGTagger", "axis2"))),
    multToken_(consumes<edm::ValueMap<int>>(edm::InputTag("QGTagger", "mult"))),
	executionMode (iConfig.getUntrackedParameter<int>("executionMode"))
{
    goodVtxNdof = iConfig.getParameter<double>("confGoodVtxNdof");
    goodVtxZ = iConfig.getParameter<double>("confGoodVtxZ");
    goodVtxRho = iConfig.getParameter<double>("confGoodVtxRho");
}

JetAnalyzer::~JetAnalyzer()
{}

void JetAnalyzer::beginJob()
{
    // Create histograms
	matchDR = fs->make<TH1D>("matchDR" , "DR of all gen/reco combinations" , 100 , 0 , 5);
	
	matchDPT = fs->make<TH1D>("matchDPT" , "recoPT/genPT of all gen/reco combinations" , 100 , 0 , 5);
	
	matchDRDPT = fs->make<TH2D>("matchDRDPT" , "DR and DPT of all gen/reco combinations" , 100 , 0 , 2, 100, 0, 5);
	matchDRDPT->GetXaxis()-> SetTitle("dR");
	matchDRDPT->GetYaxis()-> SetTitle("dPT");

        matchDRDist = fs->make<TH1D>("matchDRDist" , "DR of all selected gen/reco combinations" , 100 , 0 , 2);

	matchDPTDist = fs->make<TH1D>("matchDPTDist" , "recoPT/genPT of all selected gen/reco combinations" , 100 , 0 , 2);

	matchDRDPTDist = fs->make<TH2D>("matchDRDPTDist" , "DR vs dPT of all selected gen/reco combinations" , 100 , 0 , 0.25, 100, 0, 2);
	matchDRDPTDist->GetXaxis()-> SetTitle("dR (GeV)");
	matchDRDPTDist->GetYaxis()-> SetTitle("dPT (GeV)");

	matchDRPlusDPTDist = fs->make<TH1D>("matchDRPlusDPTDist" , "abs(1-dPT) + dR of all selected gen/reco combinations" , 100 , 0 , 5);

	matchPTdPTDist = fs->make<TH2D>("matchPTdPTDist" , "PT vs DPT of all selected gen/reco combinations" , 100 , 0 , 10, 100, 0, 2);
	matchPTdPTDist->GetXaxis()-> SetTitle("Pt (GeV)");
        matchPTdPTDist->GetYaxis()-> SetTitle("dPT (GeV)");

        genRecoPT = fs->make<TH2D>("genRecoPT" , "gen pt vs reco pt of all matched gen and reco jets" , 100 , 0 , 300, 100, 0, 300);
	genRecoPT->GetXaxis()-> SetTitle("gen Pt (GeV)");
        genRecoPT->GetYaxis()-> SetTitle("reco Pt (GeV)");

        genDR = fs->make<TH1D>("genDR" , "DR of all gen particles to gen jets" , 100 , 0 , 2);

	genDPhi = fs->make<TH1D>("genDPhi" , "DPhi of all gen particles to gen jets" , 100 , 0 , 2);

	genDEta = fs->make<TH1D>("genDEta" , "DEta of all gen particles to gen jets" , 100 , 0 , 2);

	genPT = fs->make<TH1D>("genPT" , "PT of all gen particles to gen jets" , 100 , 0 , 50);
	
	matchPercent = fs->make<TH1D>("matchPercent" , "percent of all gen jet particles which match something" , 100 , 0 , 2);

	matchNumber = fs->make<TH1D>("matchNumber" , "distribution of number of matches per jet particle" , 21 , 0 , 20);

	genNumber = fs->make<TH1D>("genNumber" , "distribution of number of particles per gen Jet" , 100 , 0 , 200);
	
	
	genJetPT = fs->make<TH1D>("genJetPT" , "PT of all jets" , 100 , 0 , 1000);
	recoJetPT = fs->make<TH1D>("recoJetPT" , "PT of all jets" , 100 , 0 , 1000);
	algJetPT = fs->make<TH1D>("algJetPT" , "PT of all jets" , 100 , 0 , 1000);
	fullAlgJetPT = fs->make<TH1D>("fullAlgJetPT" , "PT of all jets" , 100 , 0 , 1000);
	
	genJetPT0_50 = fs->make<TH1D>("genJetPT0_50" , "PT of jets with uncorrected reco PT between 0 and 25" , 100 , 0 , 300);
	recoJetPT0_50 = fs->make<TH1D>("recoJetPT0_50" , "PT of jets with uncorrected reco PT between 0 and 25" , 100 , 0 , 300);
	algJetPT0_50 = fs->make<TH1D>("algJetPT0_50" , "PT of jets with uncorrected reco PT between 0 and 25" , 100 , 0 , 300);
	fullAlgJetPT0_50 = fs->make<TH1D>("fullAlgJetPT0_50" , "PT of jets with uncorrected reco PT between 0 and 25" , 100 , 0 , 300);
	
	genJetPT50_100 = fs->make<TH1D>("genJetPT50_100" , "PT of jets with uncorrected reco PT between 50 and 55" , 100 , 0 , 300);
	recoJetPT50_100 = fs->make<TH1D>("recoJetPT50_100" , "PT of jets with uncorrected reco PT between 50 and 55" , 100 , 0 , 300);
	algJetPT50_100 = fs->make<TH1D>("algJetPT50_100" , "PT of jets with uncorrected reco PT between 50 and 55" , 100 , 0 , 300);
	fullAlgJetPT50_100 = fs->make<TH1D>("fullAlgJetPT50_100" , "PT of jets with uncorrected reco PT between 50 and 55" , 100 , 0 , 300);
	
	genJetPT100_150 = fs->make<TH1D>("genJetPT100_150" , "PT of jets with uncorrected reco PT between 100 and 105" , 100 , 0 , 300);
	recoJetPT100_150 = fs->make<TH1D>("recoJetPT100_150" , "PT of jets with uncorrected reco PT between 100 and 105" , 100 , 0 , 300);
	algJetPT100_150 = fs->make<TH1D>("algJetPT100_150" , "PT of jets with uncorrected reco PT between 100 and 105" , 100 , 0 , 300);
	fullAlgJetPT100_150 = fs->make<TH1D>("fullAlgJetPT100_150" , "PT of jets with uncorrected reco PT between 100 and 105" , 100 , 0 , 300);
	
	genJetPT150_200 = fs->make<TH1D>("genJetPT150_200" , "PT of jets with uncorrected reco PT between 150 and 155" , 100 , 0 , 300);
	recoJetPT150_200 = fs->make<TH1D>("recoJetPT150_200" , "PT of jets with uncorrected reco PT between 150 and 155" , 100 , 0 , 300);
	algJetPT150_200 = fs->make<TH1D>("algJetPT150_200" , "PT of jets with uncorrected reco PT between 150 and 155" , 100 , 0 , 300);
	fullAlgJetPT150_200 = fs->make<TH1D>("fullAlgJetPT150_200" , "PT of jets with uncorrected reco PT between 150 and 155" , 100 , 0 , 300);
	
	

	correctParticlesInReco=0;
	correctParticlesNotInReco=0;
	incorrectParticlesInReco=0;
	incorrectParticlesNotInReco=0;
}

void JetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    edm::Handle<reco::VertexCollection> vertices;
    iEvent.getByToken(vtxToken_, vertices);
    edm::Handle<pat::JetCollection> jets;
    iEvent.getByToken(jetToken_, jets);
    edm::Handle<pat::PackedCandidateCollection> pfs;
    iEvent.getByToken(pfToken_, pfs);
    edm::Handle<reco::GenJetCollection> genJets;
    iEvent.getByToken(EDMGenJetsToken_, genJets);
    edm::Handle<GenEventInfoProduct> genEventInfo;
    iEvent.getByToken(genEventInfoToken_, genEventInfo);
    edm::Handle<std::vector< PileupSummaryInfo >>  puInfo;
    iEvent.getByToken(pileupInfoToken_, puInfo);

    edm::Handle<double> pfRhoAllHandle;
    iEvent.getByToken(pfRhoAllToken_, pfRhoAllHandle);
    edm::Handle<double> pfRhoCentralHandle;
    iEvent.getByToken(pfRhoCentralToken_, pfRhoCentralHandle);
    edm::Handle<double> pfRhoCentralNeutralHandle;
    iEvent.getByToken(pfRhoCentralNeutralToken_, pfRhoCentralNeutralHandle);
    edm::Handle<double> pfRhoCentralChargedPileUpHandle;
    iEvent.getByToken(pfRhoCentralChargedPileUpToken_, pfRhoCentralChargedPileUpHandle);

    edm::Handle<edm::ValueMap<float>> qglHandle;
    iEvent.getByToken(qglToken_, qglHandle);
    edm::Handle<edm::ValueMap<float>> ptDHandle;
    iEvent.getByToken(ptDToken_, ptDHandle);
    edm::Handle<edm::ValueMap<float>> axis2Handle;
    iEvent.getByToken(axis2Token_, axis2Handle);
    edm::Handle<edm::ValueMap<int>> multHandle;
    iEvent.getByToken(multToken_, multHandle);

    // Create vectors for the jets
    // sortedJets include all jets of the event, while selectedJets have pT and eta cuts
    vector<JetIndexed> sortedJets;
    vector<JetIndexed> selectedJets;

    // Loop over the jets to save them to the jet vectors for pT-ordering
    //std::cout << "Here A" << std::endl;
    int iJetR = -1;
    for(pat::JetCollection::const_iterator jetIt = jets->begin(); jetIt!=jets->end(); ++jetIt) {
        const pat::Jet &jet = *jetIt;
        std::vector<std::string> jec = jet.availableJECSets();
        //for (std::vector<std::string>::const_iterator i = jec.begin(); i != jec.end(); ++i)
        //	std::cout << *i << std::endl;        

        //std::cout << std::endl;
        ++iJetR;
        sortedJets.push_back( JetIndexed( jet, iJetR) );
        // Select
        if ( (jet.pt() > 30) ) {// && (fabs(jet.eta()) < 2.5) ) {
			selectedJets.push_back( JetIndexed( jet, iJetR) );
        }
    }
    //std::cout << "Here B" << std::endl;

    // Sort the jets in pT order
    std::sort(sortedJets.begin(), sortedJets.end(), higher_pT_sort());
    std::sort(selectedJets.begin(), selectedJets.end(), higher_pT_sort());


    //math::XYZTLorentzVector rawRecoP4(0,0,0,0);
    //std::cout << "Here C" << std::endl;
    // Loop over the pT-ordered selected jets and save them to file
    for (unsigned int ptIdx = 0; ptIdx < selectedJets.size(); ++ptIdx) {
        math::XYZTLorentzVector rawRecoP4(0,0,0,0);

        //std::cout << "Here D" << std::endl;
        // Make selective cuts on the event level
        if (sortedJets.size() < 2) continue;
        //if (fabs(sortedJets[0].jet.eta()) > 2.5 || fabs(sortedJets[1].jet.eta()) > 2.5) continue;
        if (fabs(sortedJets[0].jet.pt()) < 30 || fabs(sortedJets[1].jet.pt()) < 30) continue;

        JetIndexed idxJet = selectedJets[ptIdx];
        const pat::Jet j = idxJet.jet;
        int iJetRef = idxJet.eventIndex;

        // Jet variables
        jetPt = j.pt();
        jetEta = j.eta();
        jetPhi = j.phi();
        jetMass = j.mass();
        jetArea = j.jetArea();

        jetRawPt = j.correctedJet("Uncorrected").pt();
        jetRawMass = j.correctedJet("Uncorrected").mass();

        jetChargedHadronMult = j.chargedHadronMultiplicity();
        jetNeutralHadronMult = j.neutralHadronMultiplicity();
        jetChargedMult = j.chargedMultiplicity();
        jetNeutralMult = j.neutralMultiplicity();

        jetPtOrder = ptIdx;

        // Determine jet IDs
        jetLooseID = 0;
        jetTightID = 0;

        Float_t nhf = j.neutralHadronEnergyFraction();
        Float_t nemf = j.neutralEmEnergyFraction();
        Float_t chf = j.chargedHadronEnergyFraction();
        Float_t cemf = j.chargedEmEnergyFraction();
        unsigned int numconst = j.chargedMultiplicity() + j.neutralMultiplicity();
        unsigned int chm = j.chargedMultiplicity();

        if (abs(j.eta())<=2.7 && (numconst>1 && nhf<0.99 && nemf<0.99) && ((abs(j.eta())<=2.4 && chf>0 && chm>0 && cemf<0.99) || abs(j.eta())>2.4)) {
            jetLooseID = 1;
            if (nhf<0.90 && nemf<0.90) {
                jetTightID = 1;
            }
        }
        //std::cout << "Here E" << std::endl;
        // Add variables for deltaPhi and deltaEta for the two leading jets of the event
        dPhiJetsLO = deltaPhi(sortedJets[0].jet.phi(), sortedJets[1].jet.phi());
        dEtaJetsLO = sortedJets[0].jet.eta() - sortedJets[1].jet.eta();
        //std::cout << "Here E1" << std::endl;


        // The alpha variable is the third jet's pT divided by the average of the two leading jets' pT
        alpha = 0;
        // Make sure that there are at least 3 jets in the event
        if(sortedJets.size() > 2) {
                Float_t leadingPtAvg = (sortedJets[0].jet.pt() + sortedJets[1].jet.pt()) * 0.5;
                alpha = sortedJets[2].jet.pt() / leadingPtAvg;
        }
        //std::cout << "Here E2" << std::endl;

        // Assign flavors for each jet using three different flavor definitions
        partonFlav = j.partonFlavour();
        hadronFlav = j.hadronFlavour();

        physFlav = 0;
        if (j.genParton()) physFlav = j.genParton()->pdgId();
        //std::cout << "Here E3" << std::endl;

        // For convenience, save variables distinguishing gluon, light quark and other jets
        isPartonUDS = 0;
        isPartonG = 0;
        isPartonOther = 0;
        isPhysUDS = 0;
        isPhysG = 0;
        isPhysOther = 0;

        // Physics definition for flavors
        if(abs(physFlav) == 1 || abs(physFlav) == 2 || abs(physFlav) == 3) {
            isPhysUDS = 1;
        } else if(abs(physFlav) == 21) {
            isPhysG = 1;
        } else {
            isPhysOther = 1;
        }
        //std::cout << "Here E4" << std::endl;

        // Parton definition for flavors
        if(abs(partonFlav) == 1 || abs(partonFlav) == 2 || abs(partonFlav) == 3) {
            isPartonUDS = 1;
        } else if(abs(partonFlav) == 21) {
            isPartonG = 1;
        } else {
            isPartonOther = 1;
        }
        //std::cout << "Here E5" << std::endl;

        // Quark-gluon likelihood variables
        edm::RefToBase<pat::Jet> jetRef(edm::Ref<pat::JetCollection>(jets, iJetRef));
        //jetQGl = (*qglHandle)[jetRef];
        //QG_ptD = (*ptDHandle)[jetRef];
        //QG_axis2 = (*axis2Handle)[jetRef];
        //QG_mult = (*multHandle)[jetRef];

        // Add event information to the jet-based tree
        event = iEvent.id().event();
        run = iEvent.id().run();
        lumi = iEvent.id().luminosityBlock();
        //std::cout << "Here E6" << std::endl;

        eventJetMult = selectedJets.size();
        //std::cout << "Here F" << std::endl;
        // MC variables
        pthat = -1;
        if (genEventInfo->hasBinningValues()) {
            pthat = genEventInfo->binningValues()[0];
        }
        eventWeight = genEventInfo->weight();

        // Pileup info
        std::vector<PileupSummaryInfo>::const_iterator PVI;
        Pileup_nTrueInt = -1;
        Pileup_nPU = -1;
        for(PVI = puInfo->begin(); PVI != puInfo->end(); ++PVI) {
            int BX = PVI->getBunchCrossing();
            if(BX == 0) {
                Pileup_nTrueInt = PVI->getTrueNumInteractions();
                Pileup_nPU = PVI->getPU_NumInteractions();
                continue;
            }
        }

        // Number of good primary vertices
        int vtxGood = 0;
        for(VertexCollection::const_iterator i_vtx = vertices->begin(); i_vtx != vertices->end(); i_vtx++) {
            if (!(i_vtx->isFake()) && i_vtx->ndof() > goodVtxNdof && fabs(i_vtx->z()) <= goodVtxZ && fabs(i_vtx->position().Rho()) <= goodVtxRho) {
                vtxGood++;
            }
        }
        PV_npvsGood = vtxGood;

        // Rhos
        rhoAll = *pfRhoAllHandle;
        rhoCentral = *pfRhoCentralHandle;
        rhoCentralNeutral = *pfRhoCentralNeutralHandle;
        rhoCentralChargedPileUp = *pfRhoCentralChargedPileUpHandle;

     	//std::cout << typeid(pfCands).name() std::endl;

        // Loop over the PF candidates contained inside the jet, first sorting them in pT order
        std::vector<reco::CandidatePtr> pfCands = j.daughterPtrVector();
        //std::cout << typeid(pfCands).name() << std::endl;
		
		std::sort(pfCands.begin(), pfCands.end(), [](const reco::CandidatePtr &p1, const reco::CandidatePtr &p2) {return p1->pt() > p2->pt(); });
        int njetpf = 0;

        // Create a PF map for easier matching later
        std::map<const pat::PackedCandidate*, const pat::PackedCandidate> pfMap;

        // Here the jet girth is also calculated
        jetGirth = 0;
		nPFR = 0;
        unsigned int pfCandsSize = pfCands.size();
        //std::cout << "Here G" << std::endl;
        for (unsigned int i = 0; i < pfCandsSize; ++i) {
            const pat::PackedCandidate &pf = dynamic_cast<const pat::PackedCandidate &>(*pfCands[i]);
            const pat::PackedCandidate* pfPointer = &pf;
            pfMap.insert(std::pair <const pat::PackedCandidate*, const pat::PackedCandidate> (pfPointer, pf));

			int pdgIDReco = pf.pdgId();
			//std::cout << pf.p4() << std::endl;
			//std::cout << pf.p4()*pf.puppiWeightNoLep() << std::endl;
			//std::cout << std::endl;	
			rawRecoP4 += pf.p4()*pf.puppiWeightNoLep();
			
			//std::cout << pf.px() << std::endl;
			
			PFR_matrix[nPFR] [0]= pf.pt();
			PFR_matrix[nPFR] [1] = pf.eta();
			PFR_matrix[nPFR] [2] = pf.phi();
			/*PFR_matrix[nPFR] [0]= pf.px();
			PFR_matrix[nPFR] [1] = pf.py();
			PFR_matrix[nPFR] [2] = pf.pz();*/
			PFR_matrix[nPFR] [3] = pf.p4().E();
			PFR_matrix[nPFR] [4] = pf.vx();
			PFR_matrix[nPFR] [5] = pf.vy();
			PFR_matrix[nPFR] [6] = pf.vz();
			PFR_matrix[nPFR] [7] = 0.0;
			PFR_matrix[nPFR] [8] = 0.0;
			PFR_matrix[nPFR] [9] = 0.0;
			PFR_matrix[nPFR] [10] = 0.0;
			PFR_matrix[nPFR] [11] = 0.0;
			PFR_matrix[nPFR] [12] = 0.0;
			PFR_matrix[nPFR] [13] = 0.0;
			PFR_matrix[nPFR] [14] = 0.0;
			PFR_matrix[nPFR] [15] = 0.0;
			PFR_matrix[nPFR] [16] = 0.0;
			PFR_matrix[nPFR] [17] = 0.0;
			
			if(pdgIDReco == -11){
				PFR_matrix[nPFR] [7] = 1.0;
			} else if(pdgIDReco == -13){
				PFR_matrix[nPFR] [9] = 1.0;
			} else if (pdgIDReco == -211){
				PFR_matrix[nPFR] [9] = 1.0;
			} else if (pdgIDReco == 22){
				PFR_matrix[nPFR] [10] = 1.0;
			} else if (pdgIDReco == 1){
				PFR_matrix[nPFR] [12] = 1.0;
			} else if (pdgIDReco == 0){
				PFR_matrix[nPFR] [12] = 1.0;
			} else if (pdgIDReco == 2){
				PFR_matrix[nPFR] [13] = 1.0;
			} else if (pdgIDReco == 130){
				PFR_matrix[nPFR] [14] = 1.0;
			} else if (pdgIDReco == 211){
				PFR_matrix[nPFR] [15] = 1.0;
			} else if (pdgIDReco == 13){
				PFR_matrix[nPFR] [16] = 1.0;
			} else if (pdgIDReco == 11){
				PFR_matrix[nPFR] [17] = 1.0;
			} 
			
			algorithm_data[nPFR] = &PFR_matrix[0][0] + 18*nPFR;
			algorithm_output[nPFR] = &PFR_output[0][0] + 4*nPFR;
			
			
			nPFR++;



            float dPhi = deltaPhi(pf.phi(), j.phi());
            float dY = pf.rapidity() - j.rapidity();

            jetGirth += sqrt(dY*dY + dPhi*dPhi) * pf.pt()/j.pt();
            ++njetpf;
        }
        //std::cout << rawRecoP4 << std::endl;
        jetMult = njetpf;
		int jetCount = 1;
		//int jetCount = constructJets.run(&algorithm_data[0], nPFR, &algorithm_output[0]);
		//std::cout << jetCount << " jets found from "<< nPFR << " particles" << std::endl;

		//for(int x =0; x<std::min(jetCount,5); x++){
		//	std::cout << *(algorithm_output[x]) << ", " << *(algorithm_output[x]+1) << ", " << *(algorithm_output[x]+2) << ", " << *(algorithm_output[x]+3) << std::endl;
		//}
		//std::cout << std::endl;
		//std::cout << "Reco jet: "<< j.p4() << std::endl;
		//std::cout << "Reco jet: " << j.pt() << ", " << j.eta() << ", " << j.phi() << ", "<< j.p4().E()   << std::endl;
		//std::cout << "Reco jet: " << j.px() << ", " << j.py() << ", " << j.pz() << ", "<< j.p4().E()   << std::endl;

        // Generator level jet variables and its constituents
        jetGenMatch = 0;
        genJetPt = 0;
        genJetEta = 0;
        genJetPhi = 0;
        genJetMass = 0;
		float genJetE = 0;
        int ng = 0;

		float AlgdR = 100;
		float AlgdPT = 100;
		float AlgPT = -10000;

        // Check if the jet has a matching generator level jet
        //std::cout << "Here J" << std::endl;
        if(j.genJet()) {
            jetGenMatch = 1;
            const reco::GenJet* gj = j.genJet();
			
			
			//std::cout << "Gen jet: " << gj->pt() << ", " << gj->eta() << ", " << gj->phi() << ", "<< gj->p4().E()   << std::endl;
			//std::cout << "Gen jet: " << gj->px() << ", " << gj->py() << ", " << gj->pz() << ", "<< gj->p4().E()   << std::endl;
			//std::cout << "Gen Reco dR:" << deltaR(j.eta(), j.phi(),
			//							gj->eta(), gj->phi()) << std::endl;
			//std::cout << "Gen New dR:" << deltaR(*(algorithm_output[0]+1) , *(algorithm_output[0]+2), 
			//							gj->eta(), gj->phi())  << std::endl;
			
            genRecoPT->Fill(gj->pt(), j.pt());
			genJetPt = gj->pt();
            genJetEta = gj->eta();
            genJetPhi = gj->phi();
            genJetMass = gj->mass();
			genJetE = gj->p4().E();
			
			
			float AlgdR = 100;
			float AlgdPT = 100;
			float AlgPT = -10000;
			//float AlgEta = -10000;
			//float AlgPhi = -10000;
			//float AlgE = -10000;
			bool matchedAlg = false;

			for(int x =0; x<std::min(jetCount,5); x++){
				double dR = deltaR(*(algorithm_output[x]+1), *(algorithm_output[x]+2),
											rawRecoP4.eta(),rawRecoP4.phi());
				double dPT = *(algorithm_output[x]+0)/rawRecoP4.pt();
				if(dR+abs(1-dPT) < AlgdR + abs(1-AlgdPT)){
						matchedAlg = true;
						AlgdR = deltaR(*(algorithm_output[x]+1), *(algorithm_output[x]+2),
											genJetEta, genJetPhi);
						AlgdPT = *(algorithm_output[x]+0)/genJetPt; 
						AlgPT = *(algorithm_output[x]);
						//AlgEta = *(algorithm_output[x]+1);
						//AlgPhi = *(algorithm_output[x]+2);
						//AlgE = *(algorithm_output[x]+3);
				}
				//std::cout << *(algorithm_output[x]) << ", " << *(algorithm_output[x]+1) << ", " << *(algorithm_output[x]+2) << ", " << *(algorithm_output[x]+3) << std::endl;
				
				
				
			}
			if(matchedAlg){
				
				
				
				genJetPT->Fill(genJetPt);
				recoJetPT->Fill(rawRecoP4.pt());
				algJetPT->Fill(AlgPT);
				if(rawRecoP4.pt()>=0 and rawRecoP4.pt()<25){
					genJetPT0_50->Fill(genJetPt);
					recoJetPT0_50->Fill(j.pt());
					algJetPT0_50->Fill(AlgPT);
				} else if(rawRecoP4.pt()>=50 and rawRecoP4.pt()<55){
					genJetPT50_100->Fill(genJetPt);
					recoJetPT50_100->Fill(j.pt());
					algJetPT50_100->Fill(AlgPT);
				} else if(rawRecoP4.pt()>=100 and rawRecoP4.pt()<=105){
					genJetPT100_150->Fill(genJetPt);
					recoJetPT100_150->Fill(rawRecoP4.pt());
					algJetPT100_150->Fill(AlgPT);
				} else if(rawRecoP4.pt()>=150 and rawRecoP4.pt()<=155){
					genJetPT150_200->Fill(genJetPt);
					recoJetPT150_200->Fill(j.pt());
					algJetPT150_200->Fill(AlgPT);
				}
			}
				
            // Loop over the genjet's constituents
            std::vector<const pat::PackedGenParticle*> genParticles;
            for (unsigned int i = 0; i < gj->numberOfDaughters(); ++i) {
                const pat::PackedGenParticle* genParticle = dynamic_cast<const pat::PackedGenParticle*>(gj->daughter(i));
                genParticles.push_back( genParticle );
            }

            // Sort the constituents in pT order
            std::sort(genParticles.begin(), genParticles.end(), [](const pat::PackedGenParticle* p1, const pat::PackedGenParticle* p2) {return p1->pt() > p2->pt(); });
            //std::cout << "Here K" << std::endl;
            unsigned int genParticlesSize = genParticles.size();
			genNumber->Fill(genParticlesSize);
            for (unsigned int i = 0; i != genParticlesSize; ++i) {
                const pat::PackedGenParticle* genParticle = dynamic_cast<const pat::PackedGenParticle*>(genParticles[i]);
		//if(!genParticle->fromHardProcessDecayed()) continue;
                float dEta = (genParticle->eta()-gj->eta());
                float dPhi = deltaPhi(genParticle->phi(), gj->phi());

                genJetPF_pT[ng] = genParticle->pt();
                genJetPF_dR[ng] = deltaR(gj->eta(), gj->phi(), genParticle->eta(), genParticle->phi());
                genJetPF_dTheta[ng] = std::atan2(dPhi, dEta);
				genJetPF_eta[ng] = genParticle->eta();
				genJetPF_phi[ng] = genParticle->phi();
                genJetPF_mass[ng] = genParticle->mass();
                genJetPF_id[ng] = genParticle->pdgId();
				genJetPF_Lorentz[ng] = genParticle->p4();
				genJetPF_jetLorentz[ng] = gj->p4()*0; //genParticlesSize;
				
				genDR->Fill(genJetPF_dR[ng]);
				genDEta->Fill(dEta);
				genDPhi->Fill(dPhi);
				genPT->Fill(genParticle->pt());
                ++ng;
            }
            nGenJetPF = ng;
        }
        //std::cout << "Here L" << std::endl;
        // Loop over all the PF candidates in an event and save those which are
        //  within the area of |deltaEta| < 1 & |deltaPhi| < 1 from the center of the jet
        if (!(kMaxPF < pfs->size()))
        assert(kMaxPF > pfs->size());
        int npfs = 0;
		
		//sort pf particles
		
		/*
		struct {
			bool operator()(pat::PackedCandidate &p1, pat::PackedCandidate &p2) const {   
				return p1.pt() < p2.pt();
			}   
        } customLess;*/
		
		//pfs->castObject(std::vector<pat::PackedCandidate>);
		
		std::vector<pat::PackedCandidate> pfsVector = *pfs;
		
		//const std::vector<pat::PackedGenParticle> *pfsParticleVector = dynamic_cast<std::vector<const pat::PackedGenParticle>*>(pfsVector);
		

		
		//std::cout << typeid(pfs).name() << std::endl;
		//std::cout << typeid(pfsVector).name() << std::endl;
		
		std::sort(pfsVector.begin(), pfsVector.end(), [](pat::PackedCandidate &p1, pat::PackedCandidate &p2) {return p1.pt() > p2.pt(); });
		nPFR=0;
        unsigned int pfsSize = pfsVector.size();
        for (unsigned int i = 0; i != pfsSize; ++i) {
            const pat::PackedCandidate &pf = (pfsVector)[i];
            const pat::PackedCandidate* pfPointer = &pf;
			
			
			
            
				
			float dEta = (pf.eta()-j.eta());
            float dPhi = deltaPhi(pf.phi(),j.phi());

            // Only save the PF candidates within the desired area
            if ( (fabs(dEta) > 1.0) || (fabs(dPhi) > 1.0) ) continue;
				
			
			int pdgIDReco = pf.pdgId();
				
			//std::cout << pf.pt() << std::endl;
			
			PFR_matrix[nPFR] [0]= pf.pt();
			PFR_matrix[nPFR] [1] = pf.eta();
			PFR_matrix[nPFR] [2] = pf.phi();
			PFR_matrix[nPFR] [3] = pf.p4().E();
			PFR_matrix[nPFR] [4] = pf.vx();
			PFR_matrix[nPFR] [5] = pf.vy();
			PFR_matrix[nPFR] [6] = pf.vz();
			PFR_matrix[nPFR] [7] = 0.0;
			PFR_matrix[nPFR] [8] = 0.0;
			PFR_matrix[nPFR] [9] = 0.0;
			PFR_matrix[nPFR] [10] = 0.0;
			PFR_matrix[nPFR] [11] = 0.0;
			PFR_matrix[nPFR] [12] = 0.0;
			PFR_matrix[nPFR] [13] = 0.0;
			PFR_matrix[nPFR] [14] = 0.0;
			PFR_matrix[nPFR] [15] = 0.0;
			PFR_matrix[nPFR] [16] = 0.0;
			PFR_matrix[nPFR] [17] = 0.0;
			
			if(pdgIDReco == -11){
				PFR_matrix[nPFR] [7] = 1.0;
			} else if(pdgIDReco == -13){
				PFR_matrix[nPFR] [9] = 1.0;
			} else if (pdgIDReco == -211){
				PFR_matrix[nPFR] [9] = 1.0;
			} else if (pdgIDReco == 22){
				PFR_matrix[nPFR] [10] = 1.0;
			} else if (pdgIDReco == 1){
				PFR_matrix[nPFR] [12] = 1.0;
			} else if (pdgIDReco == 0){
				PFR_matrix[nPFR] [12] = 1.0;
			} else if (pdgIDReco == 2){
				PFR_matrix[nPFR] [13] = 1.0;
			} else if (pdgIDReco == 130){
				PFR_matrix[nPFR] [14] = 1.0;
			} else if (pdgIDReco == 211){
				PFR_matrix[nPFR] [15] = 1.0;
			} else if (pdgIDReco == 13){
				PFR_matrix[nPFR] [16] = 1.0;
			} else if (pdgIDReco == 11){
				PFR_matrix[nPFR] [17] = 1.0;
			} 
			
			algorithm_data[nPFR] = &PFR_matrix[0][0] + 18*nPFR;
			algorithm_output[nPFR] = &PFR_output[0][0] + 4*nPFR;
			
			
			nPFR++;


			
			// Check if the PF was contained in the AK4 jet
            if (pfMap.count(pfPointer)) {
                PF_fromAK4Jet[npfs] = 1;

				
				
				
				
            } else {
                PF_fromAK4Jet[npfs] = 0;
			}
			
			
			//constructJets(&PFR_matrix[0][0], nPFR, &PFR_output[0][0]);
			
            PF_pT[npfs] = pf.pt();
            PF_dR[npfs] = deltaR(j.eta(), j.phi(), pf.eta(), pf.phi());
            PF_dTheta[npfs] = std::atan2(dPhi, dEta);
            PF_dPhi[npfs] = dPhi;
            PF_dEta[npfs] = dEta;
			PF_phi[npfs] = pf.phi();
            PF_eta[npfs] = pf.eta();
            PF_mass[npfs] = pf.mass();
            PF_id[npfs] = pf.pdgId();
            PF_fromPV[npfs] =  pf.fromPV();
			PF_Lorentz[npfs] = pf.p4();
			PF_vx[npfs] = pf.vx();
			PF_vy[npfs] = pf.vy();
			PF_vz[npfs] = pf.vz();
			
			
            ++npfs;
			
        }
        nPF = npfs;
		

		jetCount = 1;
		//jetCount = constructJets.run(&algorithm_data[0], nPFR, &algorithm_output[0]);
		//std::cout << jetCount << " jets found from "<< nPFR << " particles" << std::endl;

		float fullAlgdR = 100;
		float fullAlgdPT = 100;
		float fullAlgPT = -10000;
		float fullAlgEta = -10000;
		float fullAlgPhi = -10000;
		float fullAlgE = -10000;
		
		bool matchedAlg = false;

		for(int x =0; x<std::min(jetCount,5); x++){
			double dR = deltaR(*(algorithm_output[x]+1), *(algorithm_output[x]+2),
										rawRecoP4.eta(), rawRecoP4.phi());
			double dPT = *(algorithm_output[x]+0)/rawRecoP4.pt();
			if(dR+abs(1-dPT) < fullAlgdR + abs(1-fullAlgdPT)){
					matchedAlg = true;
					fullAlgdR =  deltaR(*(algorithm_output[x]+1), *(algorithm_output[x]+2),
										genJetEta, genJetPhi);
					fullAlgdPT = *(algorithm_output[x]+0)/genJetPt;
					fullAlgPT = *(algorithm_output[x]);
					fullAlgEta = *(algorithm_output[x]+1);
					fullAlgPhi = *(algorithm_output[x]+2);
					fullAlgE = *(algorithm_output[x]+3);
			}
			//std::cout << *(algorithm_output[x]) << ", " << *(algorithm_output[x]+1) << ", " << *(algorithm_output[x]+2) << ", " << *(algorithm_output[x]+3) << std::endl;
		}
		if(jetGenMatch==1 && matchedAlg){
				if(executionMode == 1){
					std::ofstream myfile;
					myfile.open ("mlData.txt", std::ios_base::app);
				
					myfile << fullAlgPT << "\t";
					myfile << fullAlgEta << "\t";
					myfile << fullAlgPhi << "\t";
					myfile << fullAlgE << "\t"; 
				
					myfile << rawRecoP4.pt() << "\t";
					myfile << rawRecoP4.eta() << "\t";
					myfile << rawRecoP4.phi() << "\t";
					myfile << rawRecoP4.M() << "\t"; 
					
					myfile << j.pt() << "\t";
					myfile << j.eta() << "\t";
					myfile << j.phi() << "\t";
					myfile << j.p4().M() << "\t"; 
					
					myfile << genJetPt << "\t";
					myfile << genJetEta << "\t";
					myfile << genJetPhi << "\t";
					myfile << genJetMass << "\n"; 
					
					myfile.close();
				}
				
				fullAlgJetPT->Fill(fullAlgPT);
				if(rawRecoP4.pt()>=0 and rawRecoP4.pt()<25){
					fullAlgJetPT0_50->Fill(fullAlgPT);
				} else if(rawRecoP4.pt()>=50 and rawRecoP4.pt()<55){
					fullAlgJetPT50_100->Fill(fullAlgPT);
				} else if(rawRecoP4.pt()>=100 and rawRecoP4.pt()<=105){
					fullAlgJetPT100_150->Fill(fullAlgPT);
				} else if(rawRecoP4.pt()>=150 and rawRecoP4.pt()<=155){
					fullAlgJetPT150_200->Fill(fullAlgPT);
				}
		}
		
		//std::cout << std::endl;
		
		
		int recoMatch[nPF];
		for(unsigned int y = 0; y < nPF; y++){
			recoMatch[y]=0;
		}
		
		std::ofstream myfile;
		if(executionMode == 0 || executionMode==2){
			myfile.open ("mlData.txt", std::ios_base::app);
		}
		bool firstMatch=true;
		for(unsigned int x = 0; x < nGenJetPF;x++){ 
			int matched = 0;
			unsigned int matchY=nPF;
			double minDR=100;
			double minDPT=100;
			UInt_t inReco = 0;
			
			for(unsigned int y = 0; y < nPF; y++){
				if(recoMatch[y]==1){
					continue;
				}
				double dR = deltaR((genJetPF_Lorentz[x]).eta(), (genJetPF_Lorentz[x]).phi(),
										(PF_Lorentz[y]).eta(), (PF_Lorentz[y]).phi());
				double dPT = PF_Lorentz[y].pt()/genJetPF_Lorentz[x].pt();
				if(dR+abs(1-dPT) < minDR + abs(1-minDPT)){
					minDR = dR;
					minDPT = dPT;
					matchY = y;
					inReco = PF_fromAK4Jet[y];
				}
				
				
				//matchDR->Fill(dR*1.0);
				//matchDPT->Fill(dPT*1.0);
				//matchDRDPT->Fill(dR*1.0, dPT*1.0);
				
				
				if(dR<0.2){
					matched++;
				}
			}
			if(inReco == 1){
				correctParticlesInReco+=1;
			} else {
				correctParticlesNotInReco+=1;
			}
			matchNumber->Fill(matched);
			if(matched > 0){
				matchPercent->Fill(1.0);
				matchDR->Fill(minDR*1.0);
                                matchDPT->Fill(minDPT*1.0);
                                matchDRDPT->Fill(minDR*1.0, minDPT*1.0);

			} else {
				matchPercent->Fill(0.0);
			}
			if(minDR<0.2){
				recoMatch[matchY]=1;
				
				matchDRDist->Fill(minDR);
				matchDPTDist->Fill(minDPT);
				matchDRPlusDPTDist->Fill(abs(1-minDPT)+minDR);
				matchDRDPTDist->Fill(minDR, minDPT);
				matchPTdPTDist->Fill(genJetPF_Lorentz[x].pt(), minDPT);
				if(executionMode == 0 || executionMode == 2){
					//std::cout << "Here A" << std::endl;
					if(firstMatch){
						myfile << 1 << "\t";
						firstMatch=false;
					} else {
						myfile << 0 << "\t";
					}
					
					myfile << genJetPF_Lorentz[x].Px() << "\t";
					myfile << genJetPF_Lorentz[x].Py() << "\t";
					myfile << genJetPF_Lorentz[x].Pz() << "\t";
					myfile << genJetPF_Lorentz[x].E() << "\t";
					myfile << genJetPF_id[x] << "\t";
					
					myfile << PF_Lorentz[matchY].Px() << "\t";
					myfile << PF_Lorentz[matchY].Py() << "\t";
					myfile << PF_Lorentz[matchY].Pz() << "\t";
					myfile << PF_Lorentz[matchY].E() << "\t";
					myfile << PF_vx[matchY] << "\t";
					myfile << PF_vy[matchY] << "\t";
					myfile << PF_vz[matchY] << "\t";
                                        
					if(executionMode == 2){
						//std::cout << "Here B" << std::endl;
						myfile << PF_id[matchY] << "\t";
						
						myfile << rawRecoP4.pt() << "\t";						
						myfile << rawRecoP4.eta() << "\t";
						myfile << rawRecoP4.phi() << "\t";
						myfile << rawRecoP4.E() << "\t";


						myfile << j.pt() << "\t";
        	                                myfile << j.eta() << "\t";
                	                        myfile << j.phi() << "\t";
                        	                myfile << j.p4().E() << "\t";
	
        	                                myfile << genJetPt << "\t";
                	                        myfile << genJetEta << "\t";
                        	                myfile << genJetPhi << "\t";
                                	        myfile << genJetE << "\n";

					} else {
						myfile << PF_id[matchY] << "\n";
					}
				}

				
			}
		}
		
		
		for(unsigned int y = 0; y < nPF; y++){
			if(recoMatch[y]==0){
				if(executionMode == 0 || executionMode == 2){
					if(firstMatch){
						myfile << 1 << "\t";
						firstMatch=false;
					} else {
						myfile << 0 << "\t";
					}
					
					myfile << 0.0 << "\t";
					myfile << 0.0 << "\t";
					myfile << 0.0 << "\t";
					myfile << 0.0 << "\t";
					myfile << 0.0 << "\t";
					
					myfile << PF_Lorentz[y].Px() << "\t";
					myfile << PF_Lorentz[y].Py() << "\t";
					myfile << PF_Lorentz[y].Pz() << "\t";
					myfile << PF_Lorentz[y].E() << "\t";
					myfile << PF_vx[y] << "\t";
					myfile << PF_vy[y] << "\t";
					myfile << PF_vz[y] << "\t";
					if(executionMode == 2){
                                                myfile << PF_id[y]	<< "\t";
                                                

						myfile << rawRecoP4.pt() << "\t";
                                                myfile << rawRecoP4.eta() << "\t";
                                                myfile << rawRecoP4.phi() << "\t";
                                                myfile << rawRecoP4.E() << "\t";


						myfile << j.pt() << "\t";
                                                myfile << j.eta() << "\t";
                                                myfile << j.phi() << "\t";
                                                myfile << j.p4().E() << "\t";

                                                myfile << genJetPt << "\t";
                                                myfile << genJetEta << "\t";
                                                myfile << genJetPhi << "\t";
                                                myfile << genJetE << "\n";

                                        } else {
                                                myfile << PF_id[y] << "\n";
                                        }

				}
				if(PF_fromAK4Jet[y] == 0){
					incorrectParticlesNotInReco+=1;
				} else {
					incorrectParticlesInReco+=1;
				}
				
			}
		}
		
		if(executionMode == 0 || executionMode == 2){
			myfile.close();
		}
        // Save the jet in the tree
        
		}
}


void JetAnalyzer::endJob() {
	unsigned long long  totalCorrect = correctParticlesInReco + correctParticlesNotInReco;
	unsigned long long  correctClassifications = correctParticlesInReco + incorrectParticlesNotInReco;
	unsigned long long  totalIncorrect = incorrectParticlesInReco + incorrectParticlesNotInReco;
	unsigned long long  totalInReco = correctParticlesInReco + incorrectParticlesInReco;
	//unsigned long long  totalNotInReco = correctParticlesNotInReco + incorrectParticlesNotInReco;
	double accuracy = double(correctClassifications)/double(totalCorrect+totalIncorrect);
	double precision = double(correctParticlesInReco)/double(totalInReco);
	double recall = double(correctParticlesInReco)/double(totalCorrect);
	double f1 = 2/(1/recall + 1/precision);
	
	std::cout << "Confusion Matrix:" << std::endl;
	std::cout << incorrectParticlesNotInReco << " | " << incorrectParticlesInReco << std::endl;
	std::cout << "_________________" << std::endl;
	std::cout << correctParticlesNotInReco << " | " << correctParticlesInReco << std::endl;
	std::cout << "Accuracy: " << accuracy << std::endl;
	std::cout << "Precision: " << precision << std::endl;
	std::cout << "Recall: " << recall << std::endl;
	std::cout << "f1: " << f1 << std::endl;
	
	
	
	
}

