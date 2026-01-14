 Baseline: Naive forecast (last value or moving average) for comparison 
 
REZULTATI: MAE for Naive Forecast: 48.45 Mbps 
```matlab
% Marrim të dhënat e demand 
demand_real = network_simulated_data_1_.demand_mbps; 
% Naive Forecast: Zhvendosim të dhënat me 1 pozicion (Shift) 
% Vlera e parë bëhet NaN sepse nuk kemi të dhënë para saj 
naive_pred = [NaN; demand_real(1:end-1)]; 
% Llogaritja e gabimit (MAE - Mean Absolute Error) 
mae_naive = mean(abs(demand_real - naive_pred), 'omitnan'); 
fprintf('MAE për Naive Forecast: %.2f Mbps\n', mae_naive);
```
• Main approach: Transformer-based model for time series forecasting 
REZULTATI: MAE for Moving Average (k=5): 35.47 Mbps 
```matlab
% Caktojmë dritaren (psh. 5 hapa kohorë) 
window_size = 5; 
% Përdorim funksionin movmean për të llogaritur mesataren rrëshqitëse 
% 'End' do të thotë marrim vetëm vlerat e kaluara (past observations) 
ma_filtered = movmean(demand_real, [window_size-1 0]); 
% E zhvendosim me 1 që të jetë parashikim për hapin tjetër 
ma_pred = [NaN; ma_filtered(1:end-1)]; 
% Llogaritja e gabimit (MAE) 
mae_ma = mean(abs(demand_real - ma_pred), 'omitnan'); 
fprintf('MAE për Moving Average (k=5): %.2f Mbps\n', mae_ma);
```
```matlab
% 1. Përgatitja e të dhënave 
data = network_simulated_data_1_.demand_mbps; 
X = (1:length(data))'; % Koha si variabël hyrëse 
Y = data; 
% Ndarja Train/Test (Marrim 200 pikat e para për trajnim) 
idxTrain = 1:200; 
idxTest = 201:300; 
% 2. Trajnimi i modelit GPR (Përdorim Kernel 'Matern' për trafik rrjeti) 
% Ky funksion është në Machine Learning Toolbox 
fprintf('Duke trajnuar modelin GPR...\n'); 
gprMdl = fitrgp(X(idxTrain), Y(idxTrain), 'KernelFunction', 'matern52', ... 
'Standardize', true); 
% 3. Parashikimi me Uncertainty (Intervalet e besimit) 
[ypred, ysd, yint] = predict(gprMdl, X(idxTest)); 
% 4. Vizualizimi i "Uncertainty Quantification" (Pika kryesore e detyrës) 
figure('Color', 'w'); 
plot(X(idxTest), Y(idxTest), 'k.', 'MarkerSize', 10); hold on; 
plot(X(idxTest), ypred, 'b-', 'LineWidth', 2); 
% Vizualizimi i fashës së pasigurisë (95% Confidence Interval) 
fill([X(idxTest); flipud(X(idxTest))], [yint(:,1); flipud(yint(:,2))], ... 
'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 
title('GPR Forecast: Uncertainty Quantification Analysis'); 
legend('Të dhënat Reale', 'Parashikimi ML', '95% Confidence Interval'); 
grid on; xlabel('Koha'); ylabel('Demand (Mbps)'); 
% 5. Analiza e Gabimit (MAE) 
mae_gpr = mean(abs(Y(idxTest) - ypred)); 
fprintf('MAE për modelin GPR: %.2f Mbps\n', mae_gpr);
```
<img width="587" height="271" alt="image" src="https://github.com/user-attachments/assets/35872cb0-1634-4d01-a456-092e2e15041d" />

This graphic shows us the connection between the GPR Forecast: Uncertainty Quantification 
Analysis and Demand (in Mbps)
In our graph: 
Blue line → demand forecast (in Mbps) 
Blue band→ 95% confidence interval 
Real points→ real demand in Mbps

```matlab
%% 1. PËRGATITJA (Preprocessing) 
data = network_simulated_data_1_; 
service_type = 'video'; 
subData = data(data.service == service_type, :); 
 
% Krijojmë variablat (Lidhja me kohën) 
X = (1:height(subData))';  
Y = subData.demand_mbps; 
 
% Ndarja 80% Train / 20% Test 
idx = floor(0.8 * length(Y)); 
X_train = X(1:idx); Y_train = Y(1:idx); 
X_test = X(idx+1:end); Y_test = Y(idx+1:end); 
 
%% 2. BASELINE MODELS (Pika: Comparative results vs. baseline) 
% Naive Forecast 
naive_pred = Y_test;  
naive_pred = [Y_train(end); Y_test(1:end-1)]; 
mae_naive = mean(abs(Y_test - naive_pred)); 
 
% Moving Average (k=5) 
ma_pred = movmean(Y, [4 0]); 
ma_test_pred = ma_pred(idx+1:end); 
mae_ma = mean(abs(Y_test - ma_test_pred)); 
 
%% 3. MAIN APPROACH: GAUSSIAN PROCESS REGRESSION (GPR) 
% Ky model zëvendëson Transformer-in duke ofruar pasiguri (Uncertainty) 
fprintf('Duke trajnuar modelin GPR...\n'); 
gprMdl = fitrgp(X_train, Y_train, 'KernelFunction', 'matern52', ... 
    'Standardize', true); 
 
% Parashikimi me Interval Besimi (Uncertainty Quantification) 
[y_pred, y_sd, y_int] = predict(gprMdl, X_test); 
 
%% 4. ANALIZA E GABIMIT (Pika: Error analysis) 
errors = Y_test - y_pred; 
mae_gpr = mean(abs(errors)); 
 
%% 5. VIZUALIZIMET (Për "What to Produce") 
figure('Color', 'w', 'Position', [100 100 1000 700]); 
% Grafik 1: Parashikimi dhe Uncertainty 
subplot(2,1,1); 
plot(X_test(1:100), Y_test(1:100), 'k', 'LineWidth', 1); hold on; 
plot(X_test(1:100), y_pred(1:100), 'r', 'LineWidth', 1.5); 
% Uncertainty Zone 
fill([X_test(1:100); flipud(X_test(1:100))], ... 
[y_int(1:100,1); flipud(y_int(1:100,2))], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 
title(['GPR Prediction vs Actual (MAE: ', num2str(mae_gpr,3), ')']); 
legend('Actual', 'GPR Forecast', '95% Confidence Interval (Uncertainty)'); 
grid on; 
% Grafik 2: Shpërndarja e Gabimeve 
subplot(2,1,2); 
histogram(errors, 20, 'FaceColor', [0.4 0.4 0.4]); 
title('Error Distribution Analysis: Where do predictions fail?'); 
xlabel('Error (Mbps)'); ylabel('Frequency'); 
%% 6. TABELA KRAHASUESE (Artifacts) 
ResultsTable = table(['Naive   '; 'MovAvg  '; 'GPR (ML)'], [mae_naive; mae_ma; mae_gpr], 
... 
'VariableNames', {'Model', 'MAE_Mbps'}); 
disp(ResultsTable); 
% Ruajtja e modelit (Pika 4.4) 
save('ML_Research_Artifacts.mat', 'gprMdl', 'ResultsTable', 'errors');
```
<img width="560" height="429" alt="image" src="https://github.com/user-attachments/assets/d61f3ecb-4036-4b12-8ec4-ea0b97da5f02" />

```matlab
% Përgatitja e variablave hyrëse  
% Po krijojmë një tabelë ku parashikojmë kërkesën bazuar në Orën dhe Prioritetin 
X_feat = [network_simulated_data_1_.hour, network_simulated_data_1_.priority]; 
Y_feat = network_simulated_data_1_.demand_mbps; 
% Trajnimi i një Ensemble Tree (Random Forest) 
model_feat = fitrensemble(X_feat, Y_feat, 'Method', 'Bag'); 
% Llogaritja e rëndësisë së variablave 
imp = predictorImportance(model_feat); 
figure; 
bar(imp); 
set(gca, 'XTickLabel', {'Ora', 'Prioriteti'}); 
title('Feature Importance: Çfarë po mëson modeli?'); 
ylabel('Ndikimi në Parashikim'); 
grid on;
``` 
<img width="581" height="274" alt="image" src="https://github.com/user-attachments/assets/349a991e-0883-4d8f-af51-b1beb9b92ef6" />

```matlab
% Marrim 24 pikat e fundit dhe parashikojmë të ardhmen "blindly"
last_idx = height(subData);
future_steps = 24;
X_future = (last_idx + 1 : last_idx + future_steps)';

% Përdorim modelin GPR që trajnuam më parë
[y_future, ~, y_conf] = predict(gprMdl, X_future);

figure;
plot(1:24, y_future, 'm-o', 'LineWidth', 2); hold on;
fill([1:24, fliplr(1:24)], [y_conf(:,1)', fliplr(y_conf(:,2)歪)], 'm', 'FaceAlpha', 0.1);
title('Parashikimi i Bandwidth për 24 Orët e Ardhshme');
xlabel('Orët në të ardhmen'); ylabel('Demand (Mbps)');
grid on;
```
<img width="772" height="360" alt="image" src="https://github.com/user-attachments/assets/68e30090-437d-431f-bff3-efb3bcce78df" />

```matlab
% A gabon modeli më shumë ditën apo natën?
error_by_hour = groupsummary(table(subData.hour(idx+1:end), errors), 'Var1', 'mean', 'errors');
figure; bar(error_by_hour.Var1, abs(error_by_hour.mean_errors));
title('Gabimi Mesatar sipas Orës (Error Analysis)');
```

<img width="912" height="438" alt="image" src="https://github.com/user-attachments/assets/21ed7f41-080d-4c5e-a983-784a04ec6cd2" />

```matlab
% Llogaritja e mbetjeve (residuals)
residuals = Y_test - y_pred;

figure('Name', 'Analiza e Gabimit Diagnostikues', 'Color', 'w');
subplot(2,1,1);
plot(residuals, 'LineWidth', 1);
yline(0, 'r--');
title('Analiza e Mbetjeve (Residuals over Time)');
ylabel('Gabimi (Mbps)');
```
<img width="1005" height="246" alt="image" src="https://github.com/user-attachments/assets/8c67107c-0d31-4f39-93a0-efabf1e62320" />

```matlab
%% ANALIZA DIAGNOSTIKUESE: HEATMAP E GABIMIT (Versioni Final)
% Ky kod llogarit performancën e modelit ML dhe vizualizon zonat e gabimit.

% 1. Përgatitja e të dhënave (Sigurohemi që janë të sakta)
data_clean = network_simulated_data_1_;
X_numeric = (1:height(data_clean))'; 
Y_demand = data_clean.demand_mbps;

% Ndarja Train/Test (80-20)
ndarja = floor(0.8 * length(Y_demand));
X_train = X_numeric(1:ndarja);
Y_train = Y_demand(1:ndarja);
X_test = X_numeric(ndarja+1:end);
Y_test = Y_demand(ndarja+1:end);

% 2. Trajnimi i modelit GPR (Machine Learning Toolbox)
fprintf('Duke trajnuar modelin... Ju lutem prisni.\n');
mdl = fitrgp(X_train, Y_train, 'KernelFunction', 'squaredexponential', 'Standardize', true);

% 3. Parashikimi dhe llogaritja e gabimit
y_predikim = predict(mdl, X_test);
gabimi_absolut = abs(Y_test - y_predikim);
	
% 4. Krijimi i tabelës për Heatmap (Lidhja me Orën dhe Shërbimin)
ora_test = data_clean.hour(ndarja+1:end);
sherbimi_test = data_clean.service(ndarja+1:end);

tbl_heatmap = table(ora_test, sherbimi_test, gabimi_absolut, ...
    'VariableNames', {'Ora', 'Sherbimi', 'Gabimi'});

% Agregimi i të dhënave (Mesatarja e gabimit për çdo grup)
stats = groupsummary(tbl_heatmap, {'Ora', 'Sherbimi'}, 'mean', 'Gabimi');

% 5. Ndërtimi i Heatmap me dizajn të avancuar
figure('Color', 'w', 'Name', 'ML Error Analysis');
h_map = heatmap(stats, 'Ora', 'Sherbimi', 'ColorVariable', 'mean_Gabimi');

% --- PERSONALIZIMI VIZUAL (XLabel dhe YLabel të qarta) ---
h_map.Title = 'Analiza e Gabimit Mesatar (MAE) sipas Kohës dhe Shërbimit';
h_map.XLabel = 'Ora e Ditës (00:00 - 23:00)';
h_map.YLabel = 'Lloji i Shërbimit në Rrjet';

% Ndryshimi i ngjyrave (Alternim vizual: 'Summer' ose 'Hot')
h_map.Colormap = summer; 
h_map.FontSize = 10;
h_map.ColorMethod = 'mean';

fprintf('Grafiku u gjenerua! Ky heatmap tregon pikat e dobëta të modelit ML.\n');
```
<img width="926" height="424" alt="image" src="https://github.com/user-attachments/assets/3355302e-a673-4fe4-8b37-8a05e100e8af" />

```matlab
%% ANALIZA E NDJESHMËRISË (SENSITIVITY ANALYSIS)
% Ky kod tregon se si prioriteti ndikon në parashikimin e modelit.

% 1. Marrim modelin e trajnuar (supozojmë se kemi fitrensemble ose fitrgp)
% Krijojmë një skenar artificial: Ora fikse (psh. 12:00), por prioriteti ndryshon nga 1 në 10
prioriteti_test = (1:10)';
ora_fikse = repmat(12, 10, 1); % Ora 12:00 për të gjitha pikat

X_simuluar = [ora_fikse, prioriteti_test];

% 2. Parashikojmë kërkesën për këtë skenar
% Shënim: Sigurohu që emrat e variablave përputhen me modelin tënd
mdl_final = fitrensemble([network_simulated_data_1_.hour, network_simulated_data_1_.priority], ...
                         network_simulated_data_1_.demand_mbps);

y_simuluar = predict(mdl_final, X_simuluar);

% 3. Vizualizimi i Trendit
figure('Color', 'w');
plot(prioriteti_test, y_simuluar, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
grid on;
title('Analiza e Vëmendjes: Si ndikon Prioriteti në Demand?');
xlabel('Niveli i Prioritetit (1 - 10)');
ylabel('Demand i Parashikuar (Mbps)');

% Shtojmë një shpjegim automatik
fprintf('Analiza përfundoi: Për çdo rritje të prioritetit, kërkesa e parashikuar ndryshon me %.2f Mbps.\n', ...
    mean(diff(y_simuluar)));
```
<img width="809" height="488" alt="image" src="https://github.com/user-attachments/assets/b766606f-bc45-44fc-b1a1-370c698aa6d2" />












