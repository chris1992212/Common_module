
把Recon.exe.manifest 从C:\PICKER\OKE\BIN64I 拷贝到D:\RECON64I\src\pnRecon_CPlus\Release
Recon.exe.manifest 最好存一个在C:\PICKER\OKE\BIN64I
因为C:\PICKER\OKE\BIN64I 原始是没有的


1.copy自己机器Matlab：“D:\Program Files\MATLAB\R2011b\bin\win64”下的所有dll
  覆盖到“D:\RECON64I\src\pnRecon_CPlus\Release”里
2.修改recon Proprty\Configuration Properties\VC++ Directories下的
  Include Directories和Library Directories下的路径为本机matlab对应的路径。
3.每次全编译程序时，D:\RECON64I\src\pnRecon_CPlus\Release下的Recon.exe.manifest会被清空
  所以，全编译运行之前，必须把原始的Recon.exe.manifest拷贝回release里，然后才可以运行。
  若没有copy回就运行了，必须再重新编译。
4.需要调用的dll需放在“D:\RECON64I\src\pnRecon_CPlus\Release”里
5.sensitivolume.sav及bodyimg.sav和noisepoints.sav放在C:\PICKER\OKE\Tmp

6.若新下载的代码:
  1) 将新下载的pnrecon_CPlus下的所有.h和.cpp文件替换到配好的环境里。
  2) 将thirdparty下的boost\Include\下的boost文件夹拷贝到include文件夹里。
     将ThirdParty\boost\x64\release下的所有文件copy到tmpdev\lib64i下
  3）MKL
     将ThirdParty\MKL\include_x64下的所有文件 copy到src\include里
     将ThirdParty\MKL\x64下的所有文件 copy到tmpdev\lib64i下，并将所有文件名添加到：
     Recon Property Pages\Configuration Properties\Linker\Input\Additinal Dependencies里，
	 并将之前存在的有关MKL的lib删掉
  4）若存在，将Pnrecon_CPlus下的MKL相关的3个.h文件删除
  5）将Source_code\Deployment\template\Setup\NeuStar\bin64i下的dll按名字排升序，
     将mkl相关之前的所有DLL复制到D:\RECON64I\src\pnRecon_CPlus\Release
  6）若编译通过了，但运行时出现fftw的相关error，则将rawdata_finalops.cpp中按下面的方式注释掉
     //Destroy FFTW plans
	//FFTW_DeleteXYZPlans();
	//FFTW_Delete_SENSE_XYZPlans();
    但需注意提交代码时不可以屏蔽这两行代码。

  7）将ThirdParty\DIXONPOP\Include下的文件copy到src\include里和src\pnRecon_CPlus里
     将ThirdParty\DIXONPOP下的WFI_mCoil_Recon.dll复制到D:\RECON64I\src\pnRecon_CPlus\Release
     将ThirdParty\DIXONPOP\x64下的WFI_mCoil_Recon.lib文件copy到tmpdev\lib64i下，
     并在Recon Property Pages\Configuration Properties\Linker\Input\Additinal Dependencies里添加上WFI_mCoil_Recon.lib
  8）将ThirdParty\ipp\include下的文件copy到src\include里
     将ThirdParty\ipp\x64下的所有lib文件copy到tmpdev\lib64i下，
     并在Recon Property Pages\Configuration Properties\Linker\Input\Additinal Dependencies里添加上所有lib
  9）用编译虚拟机编译新的代码，将Output\TmpDev中的文件复制覆盖到tmpdev中，
	 将Output\Release中新生成的exe和dll文件复制覆盖到pnRecon_CPlus\Release中
