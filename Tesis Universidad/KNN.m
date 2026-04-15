clear all
close all
clc

%% Cargar y acomodar base de datos
load('dataset_indoor.mat'); 

%Reshape 'features' para que sea 2D (193 x 16 x 5, numero de coordenadas)
img = reshape(img, size(img, 1)*size(img, 2)*size(img, 3), []);

%Reshape 'labels' para que sea 2D (3, 4816)
% lbl = labels.position';
labels_x = lbl(:,1);
labels_y = lbl(:,2);
labels_z = lbl(:,3);

%% Normalización con Zscore
imgz = zscore(img,0,"all");
img01 = rescale(img);

% Visualización de la disrtribución de los datos normalizados
%subplot(3, 1, 1);
figure;
histogram(img, 'Normalization', 'probability');
title('Datos sin normalizar');
xlabel('Valor Normalizado');
ylabel('Probabilidad');
% subplot(3, 1, 2);
figure;
histogram(img01, 'Normalization', 'probability');
title('Distribución de Datos Normalizados (0 - 1)');
xlabel('Valor Normalizado');
ylabel('Probabilidad');
% subplot(3, 1, 3);
figure;
histogram(imgz, 'Normalization', 'probability');
title('Distribución de Datos Normalizados (Z-Score)');
xlabel('Valor Normalizado');
ylabel('Probabilidad');

%% Proporción de datos para train y test
proporcion_entrenamiento = 0.7;  % 70% para entrenamiento, 30% para prueba

% Partición de datos para hold-out
particion = cvpartition(size(img, 2), 'HoldOut', 1 - proporcion_entrenamiento);
XTrain = imgz(:,~particion.test);
XTest = imgz(:,particion.test);
YTrain = labels_x(~particion.test,:);
YTest = labels_x(particion.test,:);

%% KNN
modelo = fitrtree(XTrain', YTrain);

% Predicción
Ypred = predict(modelo, XTest');

% Error cuadrático medio
mse = immse(Ypred, YTest);
rmse = sqrt(mse);
disp(['Error Cuadrático Medio (MSE): ' num2str(sqrt(mse))]);
save('knn.mat',"rmse");