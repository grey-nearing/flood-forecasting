import re

with open('multimet/timeseries_extractors/hres.py', 'r') as f:
    content = f.read()

replacement = """      if sec_num == 4 and sec_len >= _SECTION4_MIN_LEN_CATEGORY:
        cat_num_hdr = (sec[9], sec[10])
        if sec_len >= _SECTION4_MIN_LEN_SURFACE_TYPE:
          surface_type_hdr = sec[22]
      elif sec_num == 5:
        if sec_len >= 11:
          template_num = struct.unpack(">H", sec[9:11])[0]
          if template_num != 4:
            raise ValueError(f"Unsupported GRIB2 Section 5 template: {template_num} (expected 4 for IEEE float)")
      elif sec_num == 6:
        if sec_len >= 6:
          bitmap_indicator = sec[5]
          if bitmap_indicator != 255:
            raise ValueError(f"Unsupported GRIB2 Section 6 bitmap indicator: {bitmap_indicator} (expected 255)")
      elif sec_num == 7:"""

content = content.replace("""      if sec_num == 4 and sec_len >= _SECTION4_MIN_LEN_CATEGORY:
        cat_num_hdr = (sec[9], sec[10])
        if sec_len >= _SECTION4_MIN_LEN_SURFACE_TYPE:
          surface_type_hdr = sec[22]
      elif sec_num == 7:""", replacement)

with open('multimet/timeseries_extractors/hres.py', 'w') as f:
    f.write(content)

