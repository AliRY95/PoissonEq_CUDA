clc;
clearvars;

% data = load('sol_CPU.dat');
% x = data(:, 1); x = unique(x); Nx = length(x);
% y = data(:, 2); y = unique(y); Ny = length(y);
% u = data(:, 3); u = reshape(u, [Nx, Ny]);
% 
% [x, y] = ndgrid(x, y);
% uExact = @(x, y) sin(x).*cos(y);
% uExact = uExact(x, y);
% 
% fig = figure(1);
% subplot(1, 2, 1);
% contourf(x, y, u);
% box on; axis equal; axis tight; grid on;
% xlabel('x'); ylabel('y'); zlabel('u');
% colorbar('southoutside')
% subplot(1, 2, 2);
% contourf(x, y, u-uExact);
% box on; axis equal; axis tight; grid on;
% xlabel('x'); ylabel('y'); zlabel('u');
% colorbar('southoutside')
% % --------------------------------------------------------------------- %%
data = load('conv_data_CPU.dat');
Nx = data(:, 1);
Ny = data(:, 2);
ErrorCPU = data(:, 4);
TimeCPU = data(:, 5);
data = load('conv_data_GPU.dat');
ErrorGPU = data(:, 4);
TimeGPU = data(:, 5);
data = load('conv_data_GPU_v2.dat');
ErrorGPU2 = data(:, 4);
TimeGPU2 = data(:, 5);

% fig = figure(2);
% subplot(1, 3, 1);
% loglog(Nx, ErrorCPU, '-o', 'LineWidth', 1);
% hold on;
% loglog(Nx, ErrorGPU, '-o', 'LineWidth', 1);
% loglog(Nx, 5*Nx.^-2, 'k--', 'LineWidth', 1);
% % loglog(Nx, Nx.^-2, 'r--', 'LineWidth', 1);
% hold off;
% box on; axis square; grid on;
% xlabel('N'); ylabel('Max Error');
% legend("CPU", "GPU", 'Location', 'NE');
% map = get(gca,'ColorOrder');


% subplot(1, 3, 2);
% loglog(Nx, TimeCPU, '-o', 'LineWidth', 1);
% hold on;
% loglog(Nx, TimeGPU, '-o', 'LineWidth', 1);
% loglog(Nx, TimeGPU2, '-o', 'LineWidth', 1);
% box on; axis square; grid on;
% xlabel('N'); ylabel('Computing Time (s)');
% set(gca,'ColorOrder','factory');
% legend("CPU", "GPU with CudaMemCpy", "GPU without CudaMemCpy", 'Location', 'NW');
% 
% 
% % subplot(1, 3, 3);
% plot(Nx, TimeCPU./TimeGPU, '-o', 'LineWidth', 1);
% hold on;
% plot(Nx, TimeCPU./TimeGPU2, '-o', 'LineWidth', 1);
% box on; axis square; grid on;
% xlabel('N'); ylabel('Speedup');
% legend("GPU with CudaMemCpy", "GPU without CudaMemCpy", 'Location', 'NE');
% ylim([1, 13]);
% set(gca,'ColorOrder','factory');

figure(3);
data = load('conv_data_GPU_v3.dat');
ErrorGPU3 = data(:, 4);
TimeGPU3 = data(:, 5);
plot(Nx, TimeGPU2./TimeGPU3, '-o', 'LineWidth', 1);
hold on;
box on; axis square; grid on;
xlabel('N'); ylabel('Speedup');
set(gca,'ColorOrder','factory');
%% --------------------------------------------------------------------- %%