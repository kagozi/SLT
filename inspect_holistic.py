import os, subprocess, sys

subprocess.run(['apt-get', 'update', '-qq'], check=True)
subprocess.run(['apt-get', 'install', '-y', '-q', 'unrar'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'rarfile', '-q'], check=True)
import rarfile

base = '/data/hf_cache/How2Sign_Holistic/how2sign_holistic_features'

print('=== Directory tree ===')
for root, dirs, files in os.walk(base):
    level = root.replace(base, '').count(os.sep)
    print('  ' * level + os.path.basename(root) + '/')
    for f in sorted(files):
        size = os.path.getsize(os.path.join(root, f))
        print('  ' * (level + 1) + f + '  ({:.2f} GB)'.format(size / 1e9))

rar_path = base + '/test/frontal.rar'
print('\n=== Contents of test/frontal.rar (first 20 files) ===')
with rarfile.RarFile(rar_path) as rf:
    names = sorted(rf.namelist())
    total = len(names)
    for n in names[:20]:
        info = rf.getinfo(n)
        print('  {}  ({:.3f} MB)'.format(n, info.file_size / 1e6))
    print('  ... {} files total'.format(total))
