// Download a Sandbox scene package at run time. No Dream Labs assets are vendored.
// The web build's internal catalog is undocumented and its URL can change.
import {mkdir, writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';

const [robot, destination] = process.argv.slice(2);
if (!['openyam'].includes(robot) || !destination) {
  throw new Error('Usage: node sandbox_scene.mjs openyam DESTINATION');
}
const origin = 'https://platform.dreamlabs.ai';
const catalogUrl = `${origin}/sim-app/assets/catalog-kv7yROAD.js`;
const catalogSource = await fetch(catalogUrl).then(response => {
  if (!response.ok) throw new Error(`Sandbox catalog: HTTP ${response.status}`);
  return response.text();
});
await mkdir(destination, {recursive: true});
const modulePath = `${destination}/catalog.mjs`;
await writeFile(modulePath, catalogSource);
const originalFetch = globalThis.fetch;
globalThis.fetch = (url, options) => {
  let address = String(url);
  if (address.startsWith('file:')) address = `${origin}/sim-app/assets/${address.split('/').at(-1)}`;
  else if (address.startsWith('/sim-app/')) address = origin + address;
  return originalFetch(address, options);
};
const {b: load, c: catalog} = await import(pathToFileURL(modulePath));
const objects = [
  {id:'apple', pos:[.33,.27,0], yaw:0},
  {id:'orange', pos:[.28,-.30,0], yaw:0},
  {id:'banana', pos:[.16,-.4,0], yaw:1.57},
  {id:'water_bottle', pos:[.15,.40,0], yaw:0},
  {id:'bowl', pos:[.48,.1,0], yaw:0},
  {id:'plate', pos:[.48,-.2,0], yaw:0},
];
const result = await load({scene:'fruit_table', robot, objects});
await writeFile(`${destination}/scene.xml`, result.scene.xml);
for (const [relative, bytes] of Object.entries(result.scene.files)) {
  const parts = relative.split('/');
  if (parts.some(part => part === '..' || part === '.')) throw new Error('Unsafe package path');
  await mkdir(`${destination}/${parts.slice(0,-1).join('/')}`, {recursive:true});
  await writeFile(`${destination}/${relative}`, bytes);
}
await writeFile(`${destination}/robot.json`, JSON.stringify(catalog.robots[robot]));
