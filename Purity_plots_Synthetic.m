%Purity_plots_Synthetic

PM_Synthetic = readmatrix('PM_Synthetic.xlsx');
TKMGC_Synthetic = readmatrix('TKMGC_Synthetic.xlsx');
SPMKC_Synthetic = readmatrix('SPMKC_Synthetic.xlsx');

k = 1:3000;
figure;
plot(k,PM_Synthetic(:,3),'-.b','LineWidth',2);
hold on
plot(k,TKMGC_Synthetic(:,3),'-g','LineWidth',2);
plot(k,SPMKC_Synthetic(:,3),'--r','LineWidth',2);
axis tight
hold off
xlabel('no. of iterations (k)');
ylabel('Purity');
legend('OKC: 16.64s','TKMGC: 1182.42s','SPMKC: 110.84s');
title('Time vs Purity for all 3 Methods');