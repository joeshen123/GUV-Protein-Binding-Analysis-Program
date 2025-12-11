%Pure DPPC
figure(1);

[fit_Alps_DPPCpure_hypo_Lm,gof_Alps_DPPCpure_hypo_Lm]=FitLangmuir(Alps_DPPCpure_hypo_Lm,1:457,0 ,'kp','-','w');
[fit_Alps_DPPCpure_hypo_Lm_Ca,gof_Alps_DPPCpure_hypo_Lm_Ca]=FitLangmuir(Alps_DPPCpure_hypo_Lm,458:915,1,'rp','-','g');
legend('0 uM Ca2+','0 uM Ca2+','20 uM Ca2+','20 uM Ca2+','Location','best');

fitRes_Alps_DPPCpure_hypo_Lm=coeffvalues(fit_Alps_DPPCpure_hypo_Lm);
fitErr_Alps_DPPCpure_hypo_Lm=confint(fit_Alps_DPPCpure_hypo_Lm);
fitRes_Alps_DPPCpure_hypo_Lm_Ca=coeffvalues(fit_Alps_DPPCpure_hypo_Lm_Ca);
fitErr_Alps_DPPCpure_hypo_Lm_Ca=confint(fit_Alps_DPPCpure_hypo_Lm_Ca);


% Add X and Y label

ylabel('Norm. eGFP-emission on rim [a.u.]');
xlabel('Domain concentration [nM]');

