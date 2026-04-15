clear all
close all
clc

%% Cargar y acomodar base de datos
load('dataset_SNR50_indoor_21-11-16_23-11.mat'); 

%Reshape 'features' para que sea 2D (193 x 16 x 5, numero de coordenadas)
img = reshape(features, size(features, 1)*size(features, 2)*size(features, 3), []);

%Reshape 'labels' para que sea 2D (3, 4816)
lbl = labels.position';
labels_x = lbl(:,1);
labels_y = lbl(:,2);
labels_z = lbl(:,3);

%% Normalización con Zscore
img = zscore(img,0,"all");
% img = rescale(img);

%% Proporción de datos para train y test
proporcion_entrenamiento = 0.7;  % 70% para entrenamiento, 30% para prueba

% Semilla para tener siempre la misma division de datos
% semilla = 42;
% rng(semilla);

% Partición de datos para hold-out
particion = cvpartition(size(img, 2), 'HoldOut', 1 - proporcion_entrenamiento);
XTrain = img(:,~particion.test);
XTest = img(:,particion.test);
YTrain = labels_x(~particion.test,:);
YTest = labels_x(particion.test,:);

%% Regresión lineal
modelo = fitlm(XTrain, YTrain);

% Predicción
Ypred = predict(modelo, XTest');

% Error cuadrático medio
mse = immse(Ypred, YTest);
disp(['Error Cuadrático Medio (MSE): ' num2str(sqrt(mse))]);

