%H:\BaneDk\WorkFiles\WorkFiles
folder_name='H:\BaneDk\WorkFiles\WorkFiles\Images\ExamplesofMD';
jpegFiles = dir(fullfile(folder_name,'*.jpg'));
numfiles = length(jpegFiles); 
mydata = cell(1, numfiles);
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\Images\ExamplesofMD\'
for k = 1:numfiles
  mydata{k} = imread(jpegFiles(k).name); 
end
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\Coordinates\'
filename = 'Coordinates_long.txt';
Track_Coordinates(numfiles,mydata,filename)
fileID = fopen('Coordinates_long.txt','r');
Coor = fscanf(fileID,'%d %d %d %d %d %d %d %d %d %d',[10 Inf]);
fclose(fileID);
%%
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\CsvData\'
Excel_file_make_all('C_Long_data.xlsx',jpegFiles,Coor,mydata)
%%
folder_name='C:\Users\helen\Documents\Banedanmark\WorkFiles\Images\All';
jpegFiles = dir(fullfile(folder_name,'*.jpg'));
numfiles = length(jpegFiles); 
mydata = cell(1, numfiles);
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\Images\All\'
for k = 1:numfiles
  mydata{k} = imread(jpegFiles(k).name); 
end
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\Coordinates\'
filename = 'Coordinates_All.txt';
Track_Coordinates(numfiles,mydata,filename)
fileID = fopen('Coordinates_All.txt','r');
Coor = fscanf(fileID,'%d %d %d %d %d %d %d %d %d %d',[10 Inf]);
fclose(fileID);
%%
cd 'C:\Users\helen\Documents\Banedanmark\WorkFiles\CsvData\'
Excel_file_make_all('C_All_data.xlsx',jpegFiles,Coor,mydata)
