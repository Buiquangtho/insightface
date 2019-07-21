# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['arcface_recognition.py'],
             pathex=['C:\\pythonworks\\arcface_pyinstaller'],
             binaries=[],
             datas=[],
             hiddenimports=['cython', 'sklearn', 'sklearn.ensemble', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils','sklearn.utils._cython_blas'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='arcface_recognition',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='arcface_recognition')
