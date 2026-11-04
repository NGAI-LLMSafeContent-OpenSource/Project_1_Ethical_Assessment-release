import json,os,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));os.environ['GRADIO_ANALYTICS_ENABLED']='False'
from app_ui_2 import _build_seqcls_dataset
with tempfile.TemporaryDirectory() as d:
 os.chdir(d);source=root/'examples/binary.example.jsonl';output,summary=_build_seqcls_dataset(str(source),'text','label','1');rows=[json.loads(x) for x in Path(output).read_text().splitlines()];assert len(rows)==3;assert [x['label'] for x in rows]==[1,0,1];assert rows[0]['messages'][0]['content'].startswith('虚构样例甲')
 one=Path(d)/'single.jsonl';one.write_text(json.dumps({'text':'neutral input','label':1})+'\n')
 try:_build_seqcls_dataset(str(one),'text','label','1')
 except ValueError:pass
 else:raise AssertionError('single-class input should fail')
 print('real preprocessing and single-class rejection passed; no model training')
