k = 1
ex = rgb2gray(mydata{k}); 
imshow(ex)
%
m=500; % set max value
c_map=zeros(m,3);
c_map(1:50,1:3)=1; %white
c_map(50:100,1)=linspace(1,1,51); c_map(50:100,2)=linspace(1,1,51); c_map(50:100,3)=linspace(1,0,51); %yellow
c_map(100:150,1)=linspace(1,0,51); c_map(100:150,2)=1; %green
c_map(150:200,1)=linspace(0,0,51); c_map(150:200,2)=linspace(1,1,51); c_map(150:200,3)=linspace(0,1,51);
c_map(200:250,2:3)=1; %cyan
c_map(250:300,1)=linspace(0,1,51); c_map(250:300,2)=linspace(1,0.5,51); c_map(250:300,3)=linspace(1,0,51);
c_map(300:400,1)=1; c_map(300:400,2)=0.5; %orange
c_map(350:400,1)=1; c_map(350:400,2)=linspace(0.5,0,51);%orange
c_map(400:450,1)=1; %red
c_map(400:450,1)=linspace(1,0,51);
%
cg_map=zeros(80,3);
cg_map(1:80,1)=linspace(0,1,80);cg_map(1:80,2)=linspace(0,1,80);cg_map(1:80,3)=linspace(0,1,80);
%
se = strel('line', 10, 90);
I=ex(:,Coor(3,k):Coor(4,k));
I1 = imbothat(I,se);
I=ex(:,Coor(5,k):Coor(6,k));
I2 = imbothat(I,se);
c1_rgb = ind2rgb(I1,jet); c2_rgb = ind2rgb(I2,jet);
I_rgb = ind2rgb(ex,cg_map);
O = I_rgb; 
O(:,Coor(3,k):Coor(4,k),:) = (c1_rgb+O(:,Coor(3,k):Coor(4,k),:))/2;
O(:,Coor(5,k):Coor(6,k),:) = (c2_rgb+O(:,Coor(5,k):Coor(6,k),:))/2;


imshow(ex)
figure;
imshow(O); 
%
x = [Coor(3,k) Coor(3,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(4,k) Coor(4,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(5,k) Coor(5,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(6,k) Coor(6,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
colorbar; caxis([0 1]); colormap(jet);  

%% k50
subplot(1,4,1);
imshow(ex(900:1030,100:190)); set(gcf,'Position',[0.1 0.1 10 10]); title('Snippet of Track')
subplot(1,4,2);
imshow(O(900:1030,100:190,:)); set(gcf,'Position',[1 1 16100 900]); title('Bottom-hat transform (10)')
%
x = [Coor(3,k) Coor(3,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(4,k) Coor(4,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(5,k) Coor(5,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(6,k) Coor(6,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
colorbar; caxis([0 1]); colormap(jet);  

I1 = round( exp(entropyfilt(ex(:,Coor(3,k):Coor(4,k)),true(81,1)))/5);
c1_rgb = ind2rgb(I1,jet); c2_rgb = ind2rgb(I2,jet);
I_rgb = ind2rgb(ex,cg_map);
O1 = I_rgb; 
O1(:,Coor(3,k):Coor(4,k),:) = (c1_rgb+O1(:,Coor(3,k):Coor(4,k),:))/2;
%
subplot(1,4,3);
imshow(O1(900:1030,100:190,:)); set(gcf,'Position',[1 1 16100 900]); title('Entropy Filter (81,1)')
%
x = [Coor(3,k) Coor(3,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(4,k) Coor(4,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(5,k) Coor(5,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(6,k) Coor(6,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
colorbar; caxis([0 1]); colormap(jet);  

I1 = round( exp(entropyfilt(ex(:,Coor(3,k):Coor(4,k)),true(27,3)))/5);
c1_rgb = ind2rgb(I1,jet); c2_rgb = ind2rgb(I2,jet);
I_rgb = ind2rgb(ex,cg_map);
O1 = I_rgb; 
O1(:,Coor(3,k):Coor(4,k),:) = (c1_rgb+O1(:,Coor(3,k):Coor(4,k),:))/2;
%
subplot(1,4,4);
imshow(O1(900:1030,100:190,:)); set(gcf,'Position',[1 1 16100 900]); title('Entropy Filter (27,3)')
%
x = [Coor(3,k) Coor(3,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(4,k) Coor(4,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(5,k) Coor(5,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(6,k) Coor(6,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
colorbar; caxis([0 1]); colormap(jet);  

%% k1
subplot(1,4,1);
imshow(ex(870:1030,100:190)); set(gcf,'Position',[0.1 0.1 10 10]); title('Snippet of Track')
subplot(1,4,2);
imshow(O(870:1030,100:190,:)); set(gcf,'Position',[1 1 16100 900]); title('Bottom-hat transform (10)')
%
x = [Coor(3,k) Coor(3,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(4,k) Coor(4,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(5,k) Coor(5,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(6,k) Coor(6,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
colorbar; caxis([0 1]); colormap(jet);  

I1 = round( exp(entropyfilt(ex(:,Coor(3,k):Coor(4,k)),true(81,1)))/5);
c1_rgb = ind2rgb(I1,jet); c2_rgb = ind2rgb(I2,jet);
I_rgb = ind2rgb(ex,cg_map);
O1 = I_rgb; 
O1(:,Coor(3,k):Coor(4,k),:) = (c1_rgb+O1(:,Coor(3,k):Coor(4,k),:))/2;
%
subplot(1,4,3);
imshow(O1(870:1030,100:190,:)); set(gcf,'Position',[1 1 16100 900]); title('Entropy Filter (81,1)')
%
x = [Coor(3,k) Coor(3,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(4,k) Coor(4,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(5,k) Coor(5,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(6,k) Coor(6,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
colorbar; caxis([0 1]); colormap(jet);  

I1 = round( exp(entropyfilt(ex(:,Coor(3,k):Coor(4,k)),true(27,3)))/5);
c1_rgb = ind2rgb(I1,jet); c2_rgb = ind2rgb(I2,jet);
I_rgb = ind2rgb(ex,cg_map);
O1 = I_rgb; 
O1(:,Coor(3,k):Coor(4,k),:) = (c1_rgb+O1(:,Coor(3,k):Coor(4,k),:))/2;
%
subplot(1,4,4);
imshow(O1(900:1030,100:190,:)); set(gcf,'Position',[1 1 16100 900]); title('Entropy Filter (27,3)')
%
x = [Coor(3,k) Coor(3,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(4,k) Coor(4,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(5,k) Coor(5,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
x = [Coor(6,k) Coor(6,k)]; y = [0 Coor(1,k)]; pline = line(x,y); pline.Color = 'blue';
colorbar; caxis([0 1]); colormap(jet);  

