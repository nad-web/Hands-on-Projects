import {readdirSync} from 'node:fs';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const files=readdirSync(resolve(root,'assets')).filter(f=>f.endsWith('.js')).map(f=>'assets/'+f).concat(['sw.js','tools/build_docs.mjs']);
for(const file of files)execFileSync(process.execPath,['--check',resolve(root,file)]);
console.log(`PASS — ${files.length} application, worker, documentation, and service-worker scripts parse.`);
