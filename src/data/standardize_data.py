import music21
def mxl_to_midi():
    bach_mxl = os.path.join(EXTERNAL_DIR, "jsb_chorales")
    out = os.path.join(RAW_DIR, "jsb_chorales")
    if not os.path.exists(out): os.mkdir(out) 
    
    for i, file in enumerate(os.listdir(bach_mxl)):
        print(f"Converting file {i}/{len(os.listdir(bach_mxl))}")
        path = os.path.join(bach_mxl, file)
        out_path = os.path.join(out, file[:-4] + ".mid")
        
        d = music21.converter.parse(path)
        d.write('midi', fp=out_path)