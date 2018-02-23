% Akinlawon Solomon


%% Initialization
clc;clear all
%% Set options
% Number of contour points interpolated between the major landmarks.
options.ni=2;
% Length of landmark intensity profile
options.k = 10; 
% Search length (in pixels) for optimal contourpoint position, 
% in both normal directions of the contourpoint.
options.ns=3;
% Number of image resolution scales
options.nscales=2;
% Set normal contour, limit to +- m*sqrt( eigenvalue )
options.m=3;
% Number of search itterations
options.nsearch=60;
% If verbose is true all debug images will be shown.
options.verbose=false;
% The original minimal Mahanobis distance using edge gradient (true)
% or new minimal PCA parameters using the intensities. (false)
options.originalsearch=true;
%% ================Data Pre-Processing=================
imgdr='homework5data/database/trainImages';
landmarkdir='homework5data/database/markings';
imfile = dir(imgdr);imgs = {imfile(~[imfile.isdir]).name};
land_file = dir(landmarkdir); landp = {land_file(~[land_file.isdir]).name};
n = size(imgs,2)-2; %remove excel file and db file
imsubset = [1:220 231:440];
testsubset = [221:230 441:450];
TrainingData = struct;
p = struct;
pseudopts = 2;


for i=1:length(imsubset)
    
    temp = load(landp{1,imsubset(i)});
    p.x = temp.faceCoordinatesUnwarped(:,2)';
    p.y = temp.faceCoordinatesUnwarped(:,1)';
    p.n = size(p.x,2);
    I = (im2double((imread(imgs{1,imsubset(i)}))));   
    p.I = I;
    p.t = zeros(1,p.n);
    
    
    if  (mod(i-1,100)==0)
        options.verbose=true;
        figure; subplot(212)
    end
    [Vertices,Lines] = loadContourpts(p,pseudopts,options.verbose);
    
    if  (mod(i-1,100)==0)
        disp(i-1)
    subplot(211)
    imshow(p.I); hold on;
    P1=Vertices(Lines(:,1),:); P2=Vertices(Lines(:,2),:);
    plot([P1(:,2) P2(:,2)]',[P1(:,1) P2(:,1)]','-b');
    title('Annotated Landmark Points')
    drawnow;
    end
       
    options.verbose = false;

    TrainingData(i).Vertices=Vertices;
    TrainingData(i).Lines=Lines;
	TrainingData(i).I=(p.I);
end

%% Shape Model %%
% Make the Shape model, which finds the variations between contours
% in the training data sets. And makes a PCA model describing normal
% contours
[ShapeData,TrainingData]= ASM_MakeShapeModel2D(TrainingData);
  
% Show some eigenvector variations

    figure;
    for i=1:min(6,length(ShapeData.Evalues))
        xtest = ShapeData.x_mean + ShapeData.Evectors(:,i)*sqrt(ShapeData.Evalues(i))*2;
        subplot(2,3,i), hold on;
        plot(xtest(end/2+1:end),xtest(1:end/2),'r.');
        plot(ShapeData.x_mean(end/2+1:end),ShapeData.x_mean(1:end/2),'b.');
        title('Eigenvectors in the Shape Dimensional Space')
    end
   
    
%% Appearance model %%
% Make the Appearance model, which samples a intensity pixel profile/line 
% perpendicular to each contourpoint in each trainingdataset. Which is 
% used to build correlation matrices for each landmark. Which are used
% in the optimization step, to find the best fit.
AppearanceData = ASM_MakeAppearanceModel2D(TrainingData,options);

%% Test the ASM model %%
for i= 1:20
% Itest=im2double(imread('Fotos/test001.jpg'));
Itest = im2double(imread(imgs{1,(testsubset(i))}));
% Initial position offset and rotation, of the initial/mean contour
tform.offsetv=[0 0]; tform.offsetr=0; tform.offsets=300;
pos=[ShapeData.x_mean(1:end/2) ShapeData.x_mean(end/2+1:end)]; 
pos=ASM_align_data_inverse2D(pos,tform);

% Select the best starting position with the mouse
[x,y]=SelectPosition(Itest,pos(:,1),pos(:,2));
tform.offsetv=[-x -y];

% Apply the ASM model on the test image
tempp(:,:,i)=ASM_ApplyModel2D((Itest),tform,ShapeData,AppearanceData,options);
end

initvert = TrainingData(1).Vertices;
vertMean = mean(tempp,3);
flag = true;
posmean = [];

while flag
    for i = 1:20
        [~, newVertices(:,:,i)] = procrustes(initvert,tempp(:,:,i));
    end
    
    posmean = mean(newVertices,3);
    d = procrustes(initvert, posmean);
    
    if (d>10e-7)
        initvert = posmean;
    else
        flag = false;
    end
end

d=procrustes(initvert, posmean);

figure
hold on
plot(initvert(:,2), -initvert(:,1),'r')
plot(vertMean(:,2), -vertMean(:,1),'b')
plot(TrainingData(1).Vertices(:,2), -TrainingData(1).Vertices(:,1),'g')
title('Mean Shape of Reconstructed Points after Procrustes Alignment')
legend('initial mean','final mean shape',' baseline aligned shapes');
