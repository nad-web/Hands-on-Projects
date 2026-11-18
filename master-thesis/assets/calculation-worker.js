import {pairwiseEvidence} from './scientific.js';
self.onmessage=event=>{const {id,data}=event.data;try{self.postMessage({id,result:pairwiseEvidence(data)});}catch(error){self.postMessage({id,error:error.message});}};
