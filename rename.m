a ='C:\Users\Santanu\Desktop\5\';
A =dir( fullfile(a, '*.jpg') );
fileNames = { A.name };
for iFile = 1 : numel( A )
  newName = fullfile(a, sprintf( '0_%1d.jpg', iFile ) );
  movefile( fullfile(a, fileNames{ iFile }), newName );    
end