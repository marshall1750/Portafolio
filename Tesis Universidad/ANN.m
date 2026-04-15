
clear
close all
%clc

%% Cargar y acomodar base de datos
load('dataset_indoor.mat'); 

%Reshape 'features' para que sea 2D (193 x 16 x 5, numero de coordenadas)
img = reshape(img, size(img, 1)*size(img, 2)*size(img, 3), []);

%Reshape 'labels' para que sea 2D (3, 4816)

labels_x = lbl(:,1);
labels_y = lbl(:,2);
labels_z = lbl(:,3);

%% Normalización con Zscore
% img = zscore(img,0,"all");
img = rescale(img);

%% Proporción de datos para train y test
proporcion_entrenamiento = 0.7;  % 70% para entrenamiento, 30% para prueba

% Partición de datos para hold-out
particion = cvpartition(size(img, 2), 'HoldOut', 1 - proporcion_entrenamiento);
XTrain = img(:,~particion.test);
XTest = img(:,particion.test);
YTrain = labels_z(~particion.test,:);
YTest = labels_z(particion.test,:);

%% Red neuronal con dos capas ocultas
size_secuencia = size(XTrain,1);
capa_1 = size_secuencia/2; % Número de neuronas en la primera capa oculta
%capa_1 = 3860;
capa_2 = capa_1/2; % Número de neuronas en la segunda capa oculta
%capa_2 = 965;
capa_3 = capa_2/2; % Número de neuronas en la segunda capa oculta
%capa_3 = 242;
capa_4 = round(capa_3/2); % Número de neuronas en la segunda capa oculta
%capa_4 = 60;
capa_5 = round(capa_4/2); % Número de neuronas en la segunda capa oculta
%capa_4 = 60;
% capa_6 = round(capa_4/2); % Número de neuronas en la segunda capa oculta

% Arquitectura de la red neuronal
layers = [
    sequenceInputLayer(15440) % Capa de entrada
    fullyConnectedLayer(capa_1) % Primera capa oculta
    batchNormalizationLayer % Capa de normalización por lotes
    reluLayer % Función de activación ReLU
    fullyConnectedLayer(capa_2) % Segunda capa oculta
    batchNormalizationLayer % Capa de normalización por lotes
    reluLayer % Función de activación ReLU
    fullyConnectedLayer(capa_3) % Tercera capa oculta
    batchNormalizationLayer % Capa de normalización por lotes
    reluLayer % Función de activación ReLU
    fullyConnectedLayer(capa_4) % Cuarta capa oculta
    batchNormalizationLayer % Capa de normalización por lotes
    reluLayer % Función de activación ReLU
    % fullyConnectedLayer(capa_5) % Cuarta capa oculta
    % batchNormalizationLayer % Capa de normalización por lotes
    % reluLayer % Función de activación ReLU
    % dropoutLayer(0.3) % Dropout con una tasa del 30%
    fullyConnectedLayer(1) % Capa de salida
    regressionLayer % Capa de regresión
];

% Opciones de entrenamiento
options = trainingOptions("adam", ...
    MaxEpochs=100, ...
    MiniBatchSize=128, ...
    Shuffle="every-epoch",...
    InitialLearnRate = 1e-3, ...
    ValidationFrequency = 10, ...
    ValidationData={XTest, YTest'}, ...
    OutputNetwork = "best-validation-loss", ...
    Verbose=true, ...
    Plots="training-progress", ...
    ExecutionEnvironment= "auto");

% Entrenar la red neuronal
[net, puntos] = trainNetwork(XTrain, YTrain', layers, options);
training = puntos.TrainingRMSE;
validation = puntos.ValidationRMSE;

% Predicción
Ypred = predict(net, XTest);
Ypred = double(Ypred);

% Error cuadrático medio
mse = immse(Ypred', YTest);
rmse = sqrt(mse);
disp(['Error Cuadrático Medio (MSE): ' num2str(sqrt(mse))]);
save('ann3_4_CO_adam_100epocas_bs128_norm01.mat',"training","validation","rmse");