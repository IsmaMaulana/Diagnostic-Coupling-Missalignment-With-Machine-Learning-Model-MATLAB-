%% MAIN CODE - Coupling Misalignment Diagnostic 

%% 1. LOAD DATA
XN = load('1.NormalY_500RPM_-_16_Des_24_Recombined.mat', 'Data1_AI_2_Accelerometer_y');
RAWN = reshape(XN.Data1_AI_2_Accelerometer_y, 100000, 360);

XA2 = load('16.PARARELY_0,2_mm_500RPM_-_19_Des_24_Recombined.mat', 'Data1_AI_2_Accelerometer_y');
RAWA2 = reshape(XA2.Data1_AI_2_Accelerometer_y, 100000, 360);

XA4 = load('19._PARARELY_0.4_500RPM_-_20_Des_24_Recombined.mat', 'Data1_AI_2_Accelerometer_y');
RAWA4 = reshape(XA4.Data1_AI_2_Accelerometer_y, 100000, 360);

XA6 = load('22._PARARELY_0.6_500RPM_21_Des_24_Recombined.mat', 'Data1_AI_2_Accelerometer_y');
RAWA6 = reshape(XA6.Data1_AI_2_Accelerometer_y, 100000, 360);

XA8 = load('25._PARARELY_0.8_500RPM_-_21_Jan_24_Recombined.mat', 'Data1_AI_2_Accelerometer_y');
RAWA8 = reshape(XA8.Data1_AI_2_Accelerometer_y, 100000, 360);

%% 2. FEATURE EXTRACTION
% Fungsi untuk ekstraksi fitur
extractFeatures = @(data) [...
    rms(data); 
    kurtosis(data); 
    peak2peak(data); 
    skewness(data); 
    mean(data); 
    median(data); 
    std(data); 
    meanfreq(data); 
    medfreq(data); 
    powerbw(data); 
    bandpower(data)];

% Ekstraksi fitur untuk semua kelas
features = [
    extractFeatures(RAWN), ...
    extractFeatures(RAWA2), ...
    extractFeatures(RAWA4), ...
    extractFeatures(RAWA6), ...
    extractFeatures(RAWA8)
];

% Label kelas (5 kelas)
numClasses = 5;
samplesPerClass = 360;
classNames = {'Normal', 'Pararel0.2', 'Pararel0.4', 'Pararel0.6', 'Pararel0.8'};

%% 3. DISTANCE EVALUATION TECHNIQUE (DET) FEATURE SELECTION
% Konfigurasi DET
fds = struct();
fds.conditions = numClasses;    % Jumlah kelas
fds.sampling = 216;             % 60% dari 360
fds.selectedfeat = 5;           % Jumlah fitur terpilih

% Transpose fitur untuk format DET
fitur = features';

% Eksekusi DET
[list, d3] = detech(fitur, fds);

% Visualisasi hasil DET
figure;
plot(d3, '-ks', 'LineWidth', 1.3, 'MarkerFaceColor','r');
grid on;
axis([0 length(d3)+1 0 max(d3)+0.5]);
xlabel('Indeks Fitur');
ylabel('Faktor Efektivitas');
title('DET Feature Ranking');
set(gca, 'XTick', 1:length(d3));

% Pilih fitur terbaik
selectedFeatures = features(list(1:fds.selectedfeat), :); 

%% 4. PEMBAGIAN DATA TRAINING-TESTING
% Split data (60-40)
trainRatio = 0.6;
numTrain = floor(samplesPerClass * trainRatio);
numTest = samplesPerClass - numTrain;

% Inisialisasi
trainData = [];
testData = [];
trainLabels = [];
testLabels = [];

% Pembagian per kelas
for class = 1:numClasses
    startIdx = (class-1)*samplesPerClass + 1;
    endIdx = class*samplesPerClass;
    
    % Acak indeks
    randIdx = randperm(samplesPerClass);
    
    % Training data
    trainData = [trainData selectedFeatures(:, startIdx:startIdx+numTrain-1)];
    trainLabels = [trainLabels; repmat(class, numTrain, 1)];
    
    % Testing data
    testData = [testData selectedFeatures(:, startIdx+numTrain:endIdx)];
    testLabels = [testLabels; repmat(class, numTest, 1)];
end

%% 5. NORMALISASI DATA
% Normalisasi Z-score
[~, mu, sigma] = zscore(trainData');
trainDataNorm = (trainData' - mu) ./ sigma;
testDataNorm = (testData' - mu) ./ sigma;

%% 6. KLASIFIKASI DAN OPTIMASI KNN
% Optimasi jumlah tetangga
neighbors = 1:2:15;
accuracies = zeros(length(neighbors),1);

for k = 1:length(neighbors)
    tempModel = fitcknn(trainDataNorm, trainLabels,...
                       'NumNeighbors', neighbors(k),...
                       'Distance', 'euclidean',...
                       'Standardize', false);
    pred = predict(tempModel, testDataNorm);
    accuracies(k) = sum(pred == testLabels)/numel(testLabels)*100;
end

% Pilih K terbaik
[bestAcc, bestIdx] = max(accuracies);
bestK = neighbors(bestIdx);

% Training model final dengan K terbaik
finalKNN = fitcknn(trainDataNorm, trainLabels,...
                 'NumNeighbors', bestK,...
                 'Distance', 'euclidean',...
                 'Standardize', false);

% Prediksi dan evaluasi
finalPred = predict(finalKNN, testDataNorm);
finalAcc = sum(finalPred == testLabels)/numel(testLabels)*100;


%% 7. VISUALISASI HASIL
% Hitung confusion matrix untuk training dan testing
trainPred = predict(finalKNN, trainDataNorm);
cmTrain = confusionmat(trainLabels, trainPred);  % Confusion matrix training
cmTest = confusionmat(testLabels, finalPred);    % Confusion matrix testing

% Plot akurasi dan confusion matrix
figure;
plot(neighbors, accuracies, '-ro', 'MarkerFaceColor','b');
title(['Optimasi Parameter K (Akurasi Terbaik: ', num2str(bestAcc), '%)']);
xlabel('Jumlah Tetangga'); 
ylabel('Akurasi (%)'); 
grid on;

% Confusion matrix training
figure;
confusionchart(cmTrain, classNames,...
              'Title','Confusion Matrix Training',...
              'RowSummary','row-normalized',...
              'ColumnSummary','column-normalized');

% Confusion matrix testing
figure;
confusionchart(cmTest, classNames,...
              'Title','Confusion Matrix Testing',...
              'RowSummary','row-normalized',...
              'ColumnSummary','column-normalized');
%% 7. VISUALISASI PCA 3D DENGAN WARNA DAN SIMBOL BERBEDA
% Hitung PCA untuk 3 komponen utama
[coeffTrain, scoreTrain] = pca(trainDataNorm, 'NumComponents', 3);

% Proyeksi data testing ke 3 komponen PCA
scoreTest = testDataNorm * coeffTrain(:,1:3);

% Prediksi label data training
trainPred = predict(finalKNN, trainDataNorm);

% Identifikasi misklasifikasi
missTrainIdx = find(trainPred ~= trainLabels); % Indeks misklasifikasi training
missTestIdx = find(finalPred ~= testLabels);   % Indeks misklasifikasi testing

% Warna dan simbol untuk setiap kelas
colors = lines(numClasses); % Gunakan colormap 'lines'
symbols = {'o', 's', 'd', '^', 'v'}; % Simbol berbeda untuk 5 kelas

%% Plot 3D untuk Data Training
figure;
hold on;

% Plot data training yang benar terklasifikasi
for i = 1:numClasses
    classIdx = (trainLabels == i);
    scatter3(scoreTrain(classIdx,1), scoreTrain(classIdx,2), scoreTrain(classIdx,3),...
            100, colors(i,:), symbols{i}, 'filled');
end

% Tandai misklasifikasi training
scatter3(scoreTrain(missTrainIdx,1), scoreTrain(missTrainIdx,2), scoreTrain(missTrainIdx,3),...
        150, 'k', 'x', 'LineWidth', 2);

% Format plot
title('3D PCA - Data Training');
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend([classNames, 'Misclassification'], 'Location', 'bestoutside');
grid on;
view(-30, 15); % Sudut pandang awal
rotate3d on;    % Aktifkan rotasi 3D
hold off;

%% Plot 3D untuk Data Testing
figure;
hold on;

% Plot data testing yang benar terklasifikasi
for i = 1:numClasses
    classIdx = (testLabels == i);
    scatter3(scoreTest(classIdx,1), scoreTest(classIdx,2), scoreTest(classIdx,3),...
            100, colors(i,:), symbols{i}, 'filled');
end

% Tandai misklasifikasi testing
scatter3(scoreTest(missTestIdx,1), scoreTest(missTestIdx,2), scoreTest(missTestIdx,3),...
        150, 'k', 'x', 'LineWidth', 2);

% Format plot
title('3D PCA - Data Testing');
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend([classNames, 'Misclassification'], 'Location', 'bestoutside');
grid on;
view(-30, 15); % Sudut pandang awal
rotate3d on;    % Aktifkan rotasi 3D
hold off;

%Fungsi Dtech

function [list, d3] = detech(fitur,fds)
    [Fea_X, Fea_Y] = size(fitur);
    P = Fea_Y;
    C = fds.conditions;
    S = fds.sampling;
    
   for i = 1:P;                         
   for j = 1:C;                      
       d = 0;                        
       for m = 1:S;                  
           for n = 1:S;              
               if m ~= n               
                  d0 = abs(fitur((j-1)*S+m,i)-fitur((j-1)*S+n,i));
                  d = d0+d;              
               end                                        
          end                     
       end                                                     
      d1(i,j) = d/(S-1)/S;                                     
  end                                                          
  for j = 1:C;                     
       fitur1(j,i) = sum(fitur((j-1)*S+1:j*S,i))/S;
  end                                                          
%%%%%%%%%%%%%%%%%%             
    d = 0;                           
    for m = 1:C;                     
       for n = 1:C;                  
           if m ~= n                                                    
             d0 = abs(fitur1(m,i)-fitur1(n,i));
              d = d+d0;             
          end                      
       end                         
    end                            
    d2(i) = d/(C-1)/C;               
%%%%%%%%%%%%%%%%%%%%%           
end                                
d1 = d1';                            
d3 = d2./sum(d1)*C;                                 
d3 = d3';                                                                   
% selecting the parameters         
 order = sort(d3);                   
 SD = fds.selectedfeat;   %***                           
 selected_feat = zeros(Fea_X,SD);               
 list = zeros(SD,1);    
 
 j = 1;                              
 for  i = 1:P;                       
     if d3(i,1) >= order(P-SD+1,1);  
         para_f(:,j) = fitur(:,i);         
         list(j,1) = i;              
         j = j+1;                    
     end                           
 end
end

function fds = setting()
    fds = struct();
    fds.conditions = 5;
    fds.sampling = 216;
    fds.selectedfeat = 5;
end