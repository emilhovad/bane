%%
folder_name='C:\Users\helen\Documents\Banedanmark\WorkFiles\Images\105000-2';
jpegFiles = dir(fullfile(folder_name,'*.jpg'));
numfiles = length(jpegFiles); 
mydata = cell(1, numfiles);
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\Images\105000-2\'
for k = 1:numfiles 
  mydata{k} = imread(jpegFiles(k).name); 
end
%%
array = [];
for k = 1:numfiles
    str = strsplit(jpegFiles(k).name,{',','km'});
    array = [array; str2double(str(3)) k];
end
%
k_array = [];
for k=1:numfiles
    s_array = sortrows(array); s_array = [s_array; s_array(1,:)];
    if s_array(k)~=s_array(k+1) 
    k_array = [k_array s_array(k,2)];
    end
end
%
s_jpegFiles = jpegFiles(1:length(k_array))
for k = 1:length(k_array)
  sortdata{k} = mydata{k_array(k)}; 
  s_jpegFiles(k) = jpegFiles(k_array(k));
end
%%
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\Coordinates\'
filename = 'Coordinates.txt';
Track_Coordinates(length(k_array),sortdata,filename)
fileID = fopen('Coordinates.txt','r');
Coor = fscanf(fileID,'%d %d %d %d %d %d %d %d %d %d',[10 Inf]);
fclose(fileID);
%%
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\MakeFigure\'
k = 1;
subplot(1,2,1)
Prep_Figure(k,s_jpegFiles,Coor,sortdata)
subplot(1,2,2)
imshow(rgb2gray(sortdata{k}))
%%
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\CsvData\'
Excel_file_make_s('C1_R4_data.xlsx',s_jpegFiles,Coor,sortdata)
%%
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\CsvData\'
data_c = xlsread('C_R4_data.xlsx');
%Km,H?jd_R_D1,H?jd_L_D1,Side_R_D1,Side_L_D1,Corrugation_30-300mm_DepthRight,Corrugation_30-300mm_DepthLeft
%%
[corr(data_c(:,3),data_c(:,5)),corr(data_c(:,3),data_c(:,7)),corr(data_c(:,5),data_c(:,7))]
[corr(data_c(:,4),data_c(:,6)),corr(data_c(:,4),data_c(:,8)),corr(data_c(:,6),data_c(:,8))]
%%
fileID = fopen('R105000.txt','r');
data = fscanf(fileID,'%f');
data =  reshape(data,[9,5745]);
data = data';
fclose(fileID);
share_data = intersect(data_c(:,1),data(:,3));
%%
p1 = []; p2 = []; p3 = [];
for k = 1:length(share_data)
    if numel(find(share_data(k)==data_c(:,1)))==1
        p1 = [p1; find(share_data(k)==data_c(:,1)); find(share_data(k)==data_c(:,1))];
        p2 = [p2; find(share_data(k)==data(:,3));];
    else
        p1 = [p1; min(find(share_data(k)==data_c(:,1))); max(find(share_data(k)==data_c(:,1)))];
        p2 = [p2; find(share_data(k)==data(:,3));];
        %p3 = [p3; find(share_data(k)==data(:,3))];
    end
end
%% 
% Left
l = 3; p3 = [];
for k = 1:2:length(p1)
    p3 = [p3; max(data_c(p1(k),l),data_c(p1(k+1),l))];    
end    
[corr(p3,abs(data(p2,l+2))),corr(p3,abs(data(p2,l+4))),corr(p3,abs(data(p2,l+6)))]
% Right
r = 4; p4 = [];
for k = 1:2:length(p1)
    p4 = [p4; max(data_c(p1(k),r),data_c(p1(k+1),r))];    
end    
[corr(p4,abs(data(p2,r))),corr(p4,abs(data(p2,r+2))),corr(p4,abs(data(p2,r+4)))]
%%
% Left
l = 3; p3 = [];
for k = 1:2:length(p1)
    p3 = [p3; max(data_c(p1(k),l+2),data_c(p1(k+1),l+2))];    
end    
[corr(p3,abs(data(p2,l+2))),corr(p3,abs(data(p2,l+4))),corr(p3,abs(data(p2,l+6)))]
% Right
r = 4; p4 = [];
for k = 1:2:length(p1)
    p4 = [p4; max(data_c(p1(k),r+2),data_c(p1(k+1),r+2))];    
end    
[corr(p4,abs(data(p2,r))),corr(p4,abs(data(p2,r+2))),corr(p4,abs(data(p2,r+4)))]
%%
% Left
l = 3; p3 = [];
for k = 1:2:length(p1)
    p3 = [p3; max(data_c(p1(k),l+4),data_c(p1(k+1),l+4))];    
end    
[corr(p3,abs(data(p2,l+2))),corr(p3,abs(data(p2,l+4))),corr(p3,abs(data(p2,l+6)))]
% Right
r = 4; p4 = [];
for k = 1:2:length(p1)
    p4 = [p4; max(data_c(p1(k),r+4),data_c(p1(k+1),r+4))];    
end    
[corr(p4,abs(data(p2,r))),corr(p4,abs(data(p2,r+2))),corr(p4,abs(data(p2,r+4)))]
%% 
%L
subplot(3,2,1); scatter(p3,abs(data(p2,l+2)))
subplot(3,2,2); scatter(p3,abs(data(p2,l+4)))
subplot(3,2,3); scatter(p3,abs(data(p2,l+6)))
% R
subplot(3,2,4); scatter(p4,abs(data(p2,r)))
subplot(3,2,5); scatter(p4,abs(data(p2,r+2)))
subplot(3,2,6); scatter(p4,abs(data(p2,r+4)))
%%
%L
subplot(3,2,1); scatter(p3,exp(abs(data(p2,l+2))))
subplot(3,2,2); scatter(p3,exp(abs(data(p2,l+4))))
subplot(3,2,3); scatter(p3,exp(abs(data(p2,l+6))))
% R
subplot(3,2,4); scatter(p4,exp(abs(data(p2,r))))
subplot(3,2,5); scatter(p4,exp(abs(data(p2,r+2))))
subplot(3,2,6); scatter(p4,exp(abs(data(p2,r+4))))
%%
%L
subplot(3,2,1); scatter(p3,(data(p2,l+2)))
subplot(3,2,2); scatter(p3,(data(p2,l+4)))
subplot(3,2,3); scatter(p3,(data(p2,l+6)))
% R
subplot(3,2,4); scatter(p4,(data(p2,r)))
subplot(3,2,5); scatter(p4,(data(p2,r+2)))
subplot(3,2,6); scatter(p4,(data(p2,r+4)))
%%






%%
cd 'C:\Users\helen\Documents\Banedanmark\Scripts\Values\'
data_c = xlsread('C_R3_data.xlsx');
data_c1 = xlsread('C1_R3_data.xlsx');
%%
[corr(data_c(:,3),data_c1(:,3)) corr(data_c(:,4),data_c1(:,4)) corr(data_c(:,5),data_c1(:,5)) corr(data_c(:,6),data_c1(:,6)) corr(data_c(:,7),data_c1(:,7)) corr(data_c(:,8),data_c1(:,8))]
[mean(data_c1(:,3)-data_c(:,3)) mean(data_c1(:,4)-data_c(:,4)) mean(data_c1(:,5)-data_c(:,5)) mean(data_c1(:,6)-data_c(:,6)) mean(data_c1(:,7)-data_c(:,7)) mean(data_c1(:,8)-data_c(:,8))]

%% 
%cm ent entl wtl wtr bt bt g spor
m0 = [53.3 5.9 5.41 66 88 75 79 146.8 3];
m1 = [194 6.1 6.1 200 217 222 236 841.4 14];
n = 8;

% Left
l = 3; p3 = [];
for k = 1:2:length(p1)
    temp = (max(data_c(p1(k),l),data_c(p1(k+1),l))-m0(n))/(m1(n)-m0(n));
    if temp>1; temp=1; end; if temp<0; temp=0; end   
    p3 = [p3; temp];    
end    
[corr(p3,abs(data(p2,l+6)))]
% Right
r = 4; p4 = [];
for k = 1:2:length(p1)
    temp = (max(data_c(p1(k),r),data_c(p1(k+1),r))-m0(n))/(m1(n)-m0(n));
    if temp>1; temp=1; end; if temp<0; temp=0; end  
    p4 = [p4; temp];    
end    
[corr(p4,abs(data(p2,r+4)))]
% Left
l = 3; p3 = [];
for k = 1:2:length(p1)
    temp = (max(data_c(p1(k),l+2),data_c(p1(k+1),l+2))-m0(n))/(m1(n)-m0(n)) ;
    if temp>1; temp=1; end; if temp<0; temp=0; end   
    p3 = [p3; temp ];   
end    
[corr(p3,abs(data(p2,l+6)))]
% Right
r = 4; p4 = [];
for k = 1:2:length(p1)
      temp = (max(data_c(p1(k),r+2),data_c(p1(k+1),r+2))-m0(n))/(m1(n)-m0(n)) ;
      if temp>1; temp=1; end; if temp<0; temp=0; end   
      p4 = [p4; temp ];    
end    
[corr(p4,abs(data(p2,r+4)))]
% Left
l = 3; p3 = [];
for k = 1:2:length(p1)
    temp = (max(data_c(p1(k),l+4),data_c(p1(k+1),l+4))-m0(n))/(m1(n)-m0(n));
    if temp>1; temp=1; end; if temp<0; temp=0; end   
    p3 = [p3; temp]; 
end    
[corr(p3,abs(data(p2,l+6)))]
% Right
r = 4; p4 = [];
for k = 1:2:length(p1)
      temp = (max(data_c(p1(k),r+4),data_c(p1(k+1),r+4))-m0(n))/(m1(n)-m0(n));
      if temp>1; temp=1; end; if temp<0; temp=0; end   
      p4 = [p4; temp];    
end    
[corr(p4,abs(data(p2,r+4)))]
%% 
% Left
l = 3; p3 = [];
for k = 1:2:length(p1)
    p3 = [p3; max(data_c(p1(k),l),data_c(p1(k+1),l))];    
end    
[corr(p3,abs(data(p2,l+2))),corr(p3,abs(data(p2,l+4))),corr(p3,abs(data(p2,l+6)))]
%%
l = 3; p3 = [];
for k = 1:2:length(p1)
    p3 = [p3; max(data_c(p1(k),l+1),data_c(p1(k+1),l+1))];    
end    
[corr(p3,abs(data(p2,l+2))),corr(p3,abs(data(p2,l+4))),corr(p3,abs(data(p2,l+6)))]
%%
% Right
r = 4; p4 = [];
for k = 1:2:length(p1)
    p4 = [p4; max(data_c(p1(k),r+1),data_c(p1(k+1),r+1))];    
end    
[corr(p4,abs(data(p2,r))),corr(p4,abs(data(p2,r+2))),corr(p4,abs(data(p2,r+4)))]
%%
r = 4; p4 = [];
for k = 1:2:length(p1)
    p4 = [p4; max(data_c(p1(k),r+2),data_c(p1(k+1),r+2))];    
end    
[corr(p4,abs(data(p2,r))),corr(p4,abs(data(p2,r+2))),corr(p4,abs(data(p2,r+4)))]


%%


cd 'C:\Users\helen\Documents\Banedanmark\Scripts\Values\'
data_c = xlsread('C_R1_data.xlsx');
data_c1 = xlsread('C_R2_data.xlsx');
data_c2 = xlsread('C_R3_data.xlsx');
data_c3 = xlsread('C_R4_data.xlsx');
%%
%[corr(data_c(:,3),data_c1(:,3)) corr(data_c(:,4),data_c1(:,4)) corr(data_c(:,5),data_c1(:,5)) corr(data_c(:,6),data_c1(:,6)) corr(data_c(:,7),data_c1(:,7)) corr(data_c(:,8),data_c1(:,8))]
l = 4;
Cm = data_c(:,l); Bthr = data_c(:,l+2); Wthr = data_c(:,l+4);
Ent = data_c1(:,l); Bthl = data_c1(:,l+2); Wthl = data_c1(:,l+4);
Gab = data_c2(:,l); S1 = data_c3(:,5); S2 = data_c3(:,6);
Sam = [Cm Bthr Wthr Ent Wthl Bthl Gab max(S1,S2)];

%cm btr wtr ent bt wt g
m0 = [53.3 79 88 5.4 75 66 146.8 3];
m1 = [194 236 217 6.1 222 200 841.4 14];
SamN = Sam;

for i = 1:size(Sam,1)
for j = 1:size(Sam,2)
temp = (Sam(i,j)-m0(j))/(m1(j)-m0(j)); 
if temp>1; temp=1; end; if temp<0; temp=0; end   
SamN(i,j) = temp;
end
end
%% cm btr wtr ent bt wt g spor
dc = max(SamN')'; l = 3; p3 = []; dc = max(SamN(:,7),SamN(:,8));

dc = SamN(:,8);
for k = 1:2:length(p1)
    p3 = [p3; max(dc(p1(k)),dc(p1(k+1)))];    
end 
[corr(p3,abs(data(p2,l+2))),corr(p3,abs(data(p2,l+4))),corr(p3,abs(data(p2,l+6)))]
%%
dc = max(SamN'); r = 4; p4 = []; dc = max(SamN(:,1),SamN(:,8));
dc = SamN(:,8);
for k = 1:2:length(p1)
    p4 = [p4; max(dc(p1(k)),dc(p1(k+1)))];    
end    
[corr(p4,abs(data(p2,r))),corr(p4,abs(data(p2,r+2))),corr(p4,abs(data(p2,r+4)))]
%%
subplot(3,2,1); scatter(p3,abs(data(p2,l+2)))
subplot(3,2,2); scatter(p3,abs(data(p2,l+4)))
subplot(3,2,3); scatter(p3,abs(data(p2,l+6)))
% R
subplot(3,2,4); scatter(p4,abs(data(p2,r)))
subplot(3,2,5); scatter(p4,abs(data(p2,r+2)))
subplot(3,2,6); scatter(p4,abs(data(p2,r+4)))
%%
plotmatrix(Sam)
%%
for i = 1:8
    p3 = [];
    for k = 1:2:length(p1)
    p3 = [p3; max(SamN(p1(k),i),SamN(p1(k+1),i))];    
    end
subplot(3,3,i); scatter(p3,abs(data(p2,8)))
end




%%
% Left
l = 3; p3 = [];
for k = 1:2:length(p1)
    %p3 = [p3; max(data_c(p1(k),l:(l+1)),data_c(p1(k+1),l:(l+1)))];    
    temp = max(max(data_c(p1(k),l:(l+1))),max(data_c(p1(k+1),l:(l+1))));
    temp = (temp-3)/(14-3);
    if temp>1; temp=1; end; if temp<0; temp=0; end  
    p3 = [p3; temp];      
end    
[corr(p3,abs(data(p2,l+2))),corr(p3,abs(data(p2,l+4))),corr(p3,abs(data(p2,l+6)))]
% Right
r = 4; p4 = [];
for k = 1:2:length(p1)
    %p4 = [p4; max(data_c(p1(k),(r+1):(r+2)),data_c(p1(k+1),(r+1):(r+2)))];    
    temp = max(max(data_c(p1(k),(r+1):(r+2))),max(data_c(p1(k+1),(r+1):(r+2))));
    temp = (temp-3)/(14-3);
    if temp>1; temp=1; end; if temp<0; temp=0; end  
    p4 = [p4; temp];    
end    
[corr(p4,abs(data(p2,r))),corr(p4,abs(data(p2,r+2))),corr(p4,abs(data(p2,r+4)))]




%%
fileID = fopen('R105000.txt','r');
data = fscanf(fileID,'%f');
data =  reshape(data,[9,5745]);
data = data';
fclose(fileID);
share_data = intersect(data_c(:,1),data(:,3));
%%
cd 'C:\Users\helen\Documents\Banedanmark\Scripts\Values\'
data_c = xlsread('C_R1_data.xlsx');
data_c1 = xlsread('C_R2_data.xlsx');
data_c2 = xlsread('C_R3_data.xlsx');
data_c3 = xlsread('C_R4_data.xlsx');
%%
l = 4;
Cm = data_c(:,l); Bthr = data_c(:,l+2); Wthr = data_c(:,l+4);
Ent = data_c1(:,l); Bthl = data_c1(:,l+2); Wthl = data_c1(:,l+4);
Gab = data_c2(:,l); S1 = data_c3(:,3); S2 = data_c3(:,4);
Sam = [Cm Bthr Wthr Ent Wthl Bthl Gab max(S1,S2)];
%cm btr wtr ent bt wt g
m0 = [53.3 79 88 5.4 75 66 146.8 3];
m1 = [194 236 217 6.1 222 200 841.4 14];
SamN = Sam;
%
for i = 1:size(Sam,1)
for j = 1:size(Sam,2)
temp = (Sam(i,j)-m0(j))/(m1(j)-m0(j)); 
if temp>1; temp=1; end; if temp<0; temp=0; end   
SamN(i,j) = temp;
end
end
%%
%L 5 7 9
%R 4 6 8
p3 = [];
for k = 1:2:length(p1)
p3 = [p3; max(SamN(p1(k),1),SamN(p1(k+1),1))];    
end
subplot(1,3,1); scatter(p3,abs(data(p2,5)))
subplot(1,3,2); scatter(p3,abs(data(p2,7))) %s
subplot(1,3,3); scatter(p3,abs(data(p2,9)))
%%
subplot(1,2,1); ex = rgb2gray(sortdata{28}); imshow(ex(800:1200,:))
subplot(1,2,2); ex = imread('28.jpg'); imshow(ex(round(33+800*848/1272):round(33+1200*848/1272),92:477,:))
%%
subplot(1,2,1); ex = rgb2gray(sortdata{71}); imshow(ex(800:1200,:))
subplot(1,2,2); ex = imread('71.jpg'); imshow(ex(round(33+800*848/1272):round(33+1200*848/1272),92:477,:))
%%
subplot(1,2,1); ex = rgb2gray(sortdata{188}); imshow(ex(800:1200,:))
subplot(1,2,2); ex = imread('188.jpg'); imshow(ex(round(33+800*848/1272):round(33+1200*848/1272),92:477,:))
%%
h1 = [95 390];
h2 = [180 475];
subplot(1,6,1); ex = rgb2gray(sortdata{79}); imshow(ex(700:1000,h1(1):h2(1)))
subplot(1,6,2); ex = rgb2gray(sortdata{16}); imshow(ex(900:1200,h1(1):h2(1)))
subplot(1,6,3); ex = rgb2gray(sortdata{60}); imshow(ex(600:900,h1(1):h2(1)))
% R
subplot(1,6,4); ex = rgb2gray(sortdata{35}); imshow(ex(600:900,h1(2):h2(2)))
subplot(1,6,5); ex = rgb2gray(sortdata{160}); imshow(ex(900:1200,h1(2):h2(2)))
subplot(1,6,6); ex = rgb2gray(sortdata{180}); imshow(ex(600:900,h1(2):h2(2)))
%%
figure
subplot(1,6,1); ex = imread('79.jpg'); imshow(ex(round(33+700*848/1272):round(33+1000*848/1272),(92+95*0.6661):92+(180*0.6661),:))
subplot(1,6,2); ex = imread('16.jpg'); imshow(ex(round(33+900*848/1272):round(33+1200*848/1272),(92+95*0.6661):92+(180*0.6661),:))
subplot(1,6,3); ex = imread('60.jpg'); imshow(ex(round(33+600*848/1272):round(33+900*848/1272),(92+95*0.6661):92+(180*0.6661),:))
%
subplot(1,6,4); ex = imread('35.jpg'); imshow(ex(round(33+600*848/1272):round(33+900*848/1272),(92+390*0.6661):92+(475*0.6661),:))
subplot(1,6,5); ex = imread('160.jpg'); imshow(ex(round(33+900*848/1272):round(33+1200*848/1272),(92+390*0.6661):92+(475*0.6661),:))
subplot(1,6,6); ex = imread('180.jpg'); imshow(ex(round(33+600*848/1272):round(33+900*848/1272),(92+390*0.6661):92+(475*0.6661),:))
%%
h1 = [95 390];
h2 = [180 475];
subplot(1,6,1); ex = rgb2gray(sortdata{79}); imshow(ex(700:1000,h1(1):h2(1)))
subplot(1,6,2); ex = rgb2gray(sortdata{16}); imshow(ex(900:1200,h1(1):h2(1)))
subplot(1,6,3); ex = rgb2gray(sortdata{60}); imshow(ex(600:900,h1(1):h2(1)))
% R
subplot(1,6,4); ex = rgb2gray(sortdata{35}); imshow(ex(600:900,h1(2):h2(2)))
subplot(1,6,5); ex = rgb2gray(sortdata{160}); imshow(ex(900:1200,h1(2):h2(2)))
subplot(1,6,6); ex = rgb2gray(sortdata{180}); imshow(ex(600:900,h1(2):h2(2)))









