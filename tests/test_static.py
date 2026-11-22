from pathlib import Path
import ast
base = Path(__file__).resolve().parents[1]
for codefile in ('engine.py','server.py'):
    ast.parse((base / codefile).read_text(encoding='utf-8'))
for f in ('static/index.html','static/styles.css','static/app.js','requirements.txt','START_WINDOWS.bat'):
    assert (base/f).is_file()
print('Core source syntax and expected files OK')
