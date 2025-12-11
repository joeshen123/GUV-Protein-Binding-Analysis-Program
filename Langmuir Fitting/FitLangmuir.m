function [fitresult,gof] = FitLangmuir(TableDat,Row,PlotWithPrevious,Style,Dash, MarkerFaceColor)



TableDat=TableDat(Row,:);

conc=TableDat.ProteinConcentrationnm;

I=TableDat.NormalizedGFPintensity;


[fitresult, gof] = createFitLM_hill(conc, I);

Uconc=unique(conc);

for i=1:numel(Uconc)
    av_I(i)=mean(TableDat.NormalizedGFPintensity(TableDat.ProteinConcentrationnm==Uconc(i)));
    err_I(i)=std(TableDat.NormalizedGFPintensity(TableDat.ProteinConcentrationnm==Uconc(i)));
end
errorbar(Uconc,av_I,err_I,Style,'MarkerFaceColor',MarkerFaceColor,'MarkerSize',15, 'LineWidth',1.2);

%,'MarkerFaceColor','k'
if PlotWithPrevious~=1
hold
end

plot(fitresult,[Style(1),Dash]);
