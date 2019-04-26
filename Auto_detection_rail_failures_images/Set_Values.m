folder_name='C:\Users\helen\Documents\Banedanmark\WorkFiles\Images\Examplesof0';
jpegFiles = dir(fullfile(folder_name,'*.jpg'));
numfiles0 = length(jpegFiles); 
mydata0 = cell(1, numfiles0);
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\Images\Examplesof0\'
for k = 1:numfiles0 
  mydata0{k} = imread(jpegFiles(k).name); 
end
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\Coordinates\'
filename = 'Coordinates_0.txt';
Track_Coordinates(numfiles0,mydata0,filename)
fileID = fopen('Coordinates_0.txt','r');
Coor0 = fscanf(fileID,'%d %d %d %d %d %d %d %d %d %d',[10 Inf]);
fclose(fileID);
%%
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\MakeFigure\'
Make_Figure(numfiles0,mydata0,jpegFiles,Coor0)
%%
m_I1=0; s_I1=0; me_I1=0;
for k=1:numfiles0
ex = rgb2gray(mydata0{k});   
Ir = ex(:,Coor0(3,k):Coor0(4,k)); Il = ex(:,Coor0(7,k):Coor0(8,k));
%Ir = ex(:,Coor0(5,k):Coor0(6,k)); Il = ex(:,Coor0(9,k):Coor0(10,k));
%I1 = entropyfilt(Ir,true(27,3)); I2 = entropyfilt(Il,true(27,3));  
%I1 = entropyfilt(Ir,true(81,1)); I2 = entropyfilt(Il,true(81,1));  
se = strel('line', 10, 90);
%se = strel('sphere',7)
%se = strel('rectangle',[51 3])
%I1 = double(imtophat(Ir,se)); I2 = double(imtophat(Il,se));
I1 = double(imbothat(Ir,se)); I2 = double(imbothat(Il,se));
m_I1=max([max(max(I1)) max(max(I2)) m_I1]);
s_I1=max([std2(double(I1)) std2(double(I2)) s_I1]);
me_I1=max([mean2(double(I1)) mean2(double(I2)) me_I1]);
end
[m_I1 s_I1 me_I1]


%%
folder_name='C:\Users\helen\Documents\Banedanmark\Scripts\Values\Examplesof1';
jpegFiles = dir(fullfile(folder_name,'*.jpg'));
numfiles1 = length(jpegFiles); 
mydata1 = cell(1, numfiles1);
cd 'C:\Users\helen\Documents\Banedanmark\Scripts\Values\Examplesof1\'
for k = 1:numfiles1 
  mydata1{k} = imread(jpegFiles(k).name); 
end
cd 'C:\Users\helen\Documents\Banedanmark\Scripts\Values\'
filename = 'Coordinates_1.txt';
Track_Coordinates(numfiles1,mydata1,filename)
fileID = fopen('Coordinates_1.txt','r');
Coor1 = fscanf(fileID,'%d %d %d %d %d %d %d %d %d %d',[10 Inf]);
fclose(fileID);
%%
Make_Figure(numfiles1,mydata1,jpegFiles,Coor1)
%%
m_I1=0; s_I1=0; me_I1=0;
for k=1:numfiles1
ex = rgb2gray(mydata1{k});   
Ir = ex(:,Coor1(3,k):Coor1(4,k)); Il = ex(:,Coor1(7,k):Coor1(8,k));
%I1 = entropyfilt(Ir,true(27,3)); I2 = entropyfilt(Il,true(27,3));  
%I1 = entropyfilt(Ir,true(81,1)); I2 = entropyfilt(Il,true(81,1));  
%se = strel('line', 10, 90);
%se = strel('sphere',7)
se = strel('rectangle',[51 3])
%I1 = double(imtophat(Ir,se)); I2 = double(imtophat(Il,se));
I1 = double(imbothat(Ir,se)); I2 = double(imbothat(Il,se));
m_I1=max([max(max(I1)) max(max(I2)) m_I1]);
s_I1=max([std2(double(I1)) std2(double(I2)) s_I1]);
me_I1=max([mean2(double(I1)) mean2(double(I2)) me_I1]);
end
[m_I1 s_I1 me_I1]

%%
mu = mean(Ir,1);
st = std(double(Ir),1);
I = abs(repmat(mu,size(Ir,1),1)-double(Ir))- repmat(st,size(Ir,1),1);
I1 = I.*(I>0);

mu = mean(Il,1);
st = std(double(Il),1);
I = abs(repmat(mu,size(Il,1),1)-double(Il))- repmat(st,size(Il,1),1);
I2 = I.*(I>0);

%I1 = stdfilt(Ir,true(1,3)); I2 = stdfilt(Il,true(1,3));
%I1 = rangefilt(Ir,true(1,3)); I2 = stdfilt(Il,true(1,3));
%h = fspecial('sobel')';
%I1 = imfilter(Ir,h)+imfilter(Ir,-1.*h); I2 = imfilter(Il,h)+imfilter(Il,-1.*h);
%%
k = 1
ex = rgb2gray(mydata0{k}); Ir = ex(:,Coor0(3,k):Coor0(4,k)); Il = ex(:,Coor0(7,k):Coor0(8,k));
%ex = rgb2gray(mydata1{k}); Ir = ex(:,Coor1(3,k):Coor1(4,k)); Il = ex(:,Coor1(7,k):Coor1(8,k));
%ex = rgb2gray(mydata1{k}); Ir = ex(:,Coor1(5,k):Coor1(6,k)); Il = ex(:,Coor1(9,k):Coor1(10,k));
F = fft2(imgaussfilt(Ir,2));
%A = angle(F);
F = fftshift(F); % Center FFT
F = abs(F); % Get the magnitude
F = log(F+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
F = mat2gray(F); % Use mat2gray to scale the image between 0 and 1
imshow(F,[]); % Display the result
%%
plot(mean(F,2))
%%
plot(sgolayfilt(max(F'),3,11))
[TF,P] = islocalmax(sgolayfilt(max(F'),3,11)); 
plot(P)
%%
k = 13
%ex = rgb2gray(mydata0{k}); Ir = ex(:,Coor0(3,k):Coor0(4,k)); Il = ex(:,Coor0(7,k):Coor0(8,k));
ex = rgb2gray(mydata1{k}); Ir = ex(:,Coor1(3,k):Coor1(4,k)); Il = ex(:,Coor1(7,k):Coor1(8,k));
ex = rgb2gray(mydata1{k}); Ir = ex(:,Coor1(5,k):Coor1(6,k)); Il = ex(:,Coor1(9,k):Coor1(10,k));
imshow(xcorr2(Il),[])
%imshow(Il)
%%
k = 1
ex = rgb2gray(mydata0{k}); Ir = ex(:,Coor0(3,k):Coor0(4,k)); Il = ex(:,Coor0(7,k):Coor0(8,k));
%ex = rgb2gray(mydata1{k}); Il = ex(:,Coor1(3,k):Coor1(4,k)); Ir = ex(:,Coor1(7,k):Coor1(8,k));
I = Ir;
gaborArray = gabor([4],[0 90]);
gaborMag = imgaborfilt(I,gaborArray);

figure
subplot(1,3,1);
imshow(I);
title('Original Image');
subplot(1,3,2);
imshow(gaborMag(:,:,1),[])
subplot(1,3,3);
imshow(gaborMag(:,:,2),[]);
%%
plot(max(gaborMag(:,:,2)'))
%%
m_I1=[]; s_I1=[]; me_I1=[];
for k = 1:numfiles0
ex = rgb2gray(mydata0{k}); Ir = ex(:,Coor0(3,k):Coor0(4,k)); Il = ex(:,Coor0(7,k):Coor0(8,k));
gaborArray = gabor([4],[90]); I1 = imgaborfilt(Ir,gaborArray);
gaborArray = gabor([4],[90]); I2 = imgaborfilt(Il,gaborArray);
m_I1= [m_I1 max([max(max(I1)) max(max(I2))])];
s_I1= [s_I1 max([std2(double(I1)) std2(double(I2))])];
me_I1= [me_I1 max([mean2(double(I1)) mean2(double(I2))])];
end
m_I1
[max(m_I1) max(s_I1) max(me_I1)]
%%
m_I1=[]; s_I1=[]; me_I1=[];
for k = 1:numfiles1
ex = rgb2gray(mydata1{k}); Ir = ex(:,Coor1(3,k):Coor1(4,k)); Il = ex(:,Coor1(7,k):Coor1(8,k));
gaborArray = gabor([4],[90]); I1 = imgaborfilt(Ir,gaborArray);
gaborArray = gabor([4],[90]); I2 = imgaborfilt(Il,gaborArray);
m_I1= [m_I1 max([max(max(I1)) max(max(I2))])];
s_I1= [s_I1 max([std2(double(I1)) std2(double(I2))])];
me_I1= [me_I1 max([mean2(double(I1)) mean2(double(I2))])];
end
m_I1
[max(m_I1) max(s_I1) max(me_I1)]

%%
for k = 1:numfiles0
ex = rgb2gray(mydata0{k}); Ir = ex(:,Coor0(3,k):Coor0(4,k)); Il = ex(:,Coor0(7,k):Coor0(8,k));
m_I1= [m_I1 max([max(max(I1)) max(max(I2))])];
end
%%
%%
ex = rgb2gray(mydata0{9});
imshow(ex)
%%
p1 = []; p2 = []; p3 = []; p4 = [];
for k = 1:numfiles0
for l = 1:100:(Coor0(1,k)-100)    
[r1, r2] = Sobel(Coor0,mydata0,1,l:(l+100),k);
[r3, r4] = Sobel(Coor0,mydata0,2,l:(l+100),k);
p1 = [p1 r1]; p2 = [p2 r2]; p3 = [p3 r3]; p4 = [p4 r4];
end
end
plot(p1); hold on; plot(p2); hold off;
figure
plot(p3); hold on; plot(p4); hold off;
%%
ex = rgb2gray(mydata1{3});
imshow(ex)
%%
p1 = []; p2 = []; p3 = []; p4 = [];
for k = 1:numfiles1
for l = 1:100:(Coor1(1,k)-100)    
[r1, r2] = Sobel(Coor1,mydata1,1,l:(l+100),k);
[r3, r4] = Sobel(Coor1,mydata1,2,l:(l+100),k);
p1 = [p1 r1]; p2 = [p2 r2]; p3 = [p3 r3]; p4 = [p4 r4];
end
end
plot(p1); hold on; plot(p2); hold off;
figure
plot(p3); hold on; plot(p4); hold off;




