folder_name='H:\BaneDk\WorkFiles\WorkFiles\Images\All\';
jpegFiles = dir(fullfile(folder_name,'*.jpg'));
numfiles = length(jpegFiles); 
mydata = cell(1, numfiles);
cd 'H:\BaneDk\WorkFiles\WorkFiles\Images\All\'
for k = 1:numfiles 
  mydata{k} = imread(jpegFiles(k).name); 
end
%%
cd 'H:\BaneDk\WorkFiles\WorkFiles\Coordinates\'
fileID = fopen('Coordinates_All.txt','r');
Coor = fscanf(fileID,'%d %d %d %d %d %d %d %d %d %d',[10 Inf]);
fclose(fileID);
%%
m = numfiles;
d1 = zeros(1,m); d2 = zeros(1,m); d1m = zeros(1,m); d2m = zeros(1,m);
n=4;
for k = 1:m
    ex = rgb2gray(mydata{k});   
    h = fspecial('sobel')';
    I = imfilter(ex,h)+imfilter(ex,-1.*h);
    I = imerode(I,strel('line', 40, 90));
    I = I-40;
    %I = I.*rangefilt(ex,true(1,3));    
    %I = imerode(I,strel('rectangle',[51 2]));
    %I = I-100;    
    A = [10:280].*(sum(I(1:200,10:280))>(max(sum(I(1:200,10:280)))/n));
    A = A(A ~= 0);
    B = [10:280].*(sum(I((size(I,1)-200):size(I,1),10:280))>(max(sum(I((size(I,1)-200):size(I,1),10:280)))/n));
    B = B(B ~= 0);
    d1(k) = abs(min(A)-min(B)); d1(k) = min(A);
    d1m(k) = abs(max(A)-max(B)); d1m(k) = max(A);

    A = [300:570].*(sum(I(1:200,300:570))>(max(sum(I(1:200,300:570)))/n));
    A = A(A ~= 0);
    B = [300:570].*(sum(I((size(I,1)-200):size(I,1),300:570))>(max(sum(I((size(I,1)-200):size(I,1),300:570)))/n));
    B = B(B ~= 0);
    d2(k) = abs(min(A)-min(B)); d2(k) = min(A);
    d2m(k) = abs(max(A)-max(B)); d2m(k) = max(A);
end
%%
quantile(d1,[0.05 0.25 0.50 0.75 0.95])
quantile(d1m,[0.05 0.25 0.50 0.75  0.95])
quantile(d2,[0.05 0.25 0.50 0.75  0.95])
quantile(d2m,[0.05 0.25 0.50 0.75  0.95])
%%
plot(d1)
hold on
plot(d1m+50)
plot(d2+100)
plot(d2m+150)
hold off
legend('V1','V2','H1','H2')
%%
quantile(d1,[0.25 0.50 0.75 0.9 0.85])
quantile(d1m,[0.25 0.50 0.75 0.9 0.85])
quantile(d2,[0.25 0.50 0.75 0.9 0.85])
quantile(d2m,[0.25 0.50 0.75 0.9 0.85])
%%
quantile(abs(d1-d1m),[0.05 0.25 0.50 0.75 0.95])
quantile(abs(d2-d2m),[0.05 0.25 0.50 0.75 0.95])