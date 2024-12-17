import os
import pandas as pd
import shutil
from pathlib import Path

sns = os.listdir(r'C:\PycharmProjects\SchemeRep\fMRI_in')
possible_drops = ['Bl_NoGSR_8', 'Con_NoGSR_8', 'Enc_NoGSR_8', 'Vis_NoGSR_8']
for sn in sns:
	print(f'Onto: {sn}')
	sn_dir = fr'C:\PycharmProjects\SchemeRep\fMRI_in\{sn}'
	for d in possible_drops:
		drop_dir = fr'{sn_dir}\{d}'
		if os.path.exists(drop_dir):
			Path(fr'H:\fMRI_in\{sn}\{d}').mkdir(parents=True, exist_ok=True)
			shutil.move(drop_dir, fr'H:\fMRI_in\{sn}\{d}')
			print('Moved', fr'H:\fMRI_in\{sn}\{d}')