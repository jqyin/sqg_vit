
%%
%SC24 paper Fig 4
%--------------------------------------------------------------------------
%%


clear all
close all

load Linear_shocks.mat
SQG=readmatrix("sqg_only_one_ensmble.xlsx");
Vitonly=readmatrix("rmse_ensf_initial_only.csv");
vit_Ensf=readmatrix("rmse_ensf_jump_noise.csv");

dt = 12;
T = dt * length(ntime);
   
scale_factor = 1;

figure,
width=1000;
height=600;
left= 200;
bottem=250;
set(gcf,'position',[left,bottem,width,height])
set(gcf, 'Renderer', 'painters');
hold on

plot(dt:dt:T, vit_Ensf/scale_factor, '-o', 'Color', '#EDB120', 'markersize', 4, 'linewidth', 2)
plot(dt:dt:T, Vitonly(1:300)/scale_factor, '-^ r','Color','#77AC30' ,'markersize', 4, 'linewidth', 2)
plot(dt:dt:T, LETKF/scale_factor, '-+ r', 'markersize', 4, 'linewidth', 2)

%plot(dt:dt:T, EnSF/scale_factor, '-x b', 'markersize', 4, 'linewidth', 2)

plot(dt:dt:T, SQG/scale_factor, '-d k', 'markersize', 4, 'linewidth', 2)

%legend('LETKF', 'EnSF', 'SQG', 'location', 'best','NumColumns',1)
legend('ViT+EnSF','ViT Only','SQG+LETKF', 'SQG Only', 'location', 'best','NumColumns',2)

xlabel('Time [hours]')
ylabel('RMSE [K]')

ylim([0 14.5])
xlim([0 3600])

set(gca, 'FontName', 'DejaVu Sans', 'FontSize', 18)
exportgraphics(gcf, 'plot.png', 'Resolution', 1200);
hold off
