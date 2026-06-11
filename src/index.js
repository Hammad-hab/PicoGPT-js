import GPT2 from "./Pico.js"
import * as tf from "@tensorflow/tfjs"
import * as Util from "./Utils.js"
import { question } from "readline-sync"

const generateToken = (inputs, params, nHead) => {
    const logits = GPT2(inputs, params['wte'], params['wpe'], params['blocks'], params['ln_f'], nHead)
    const nextId = tf.argMax(logits, 1).dataSync()[0]
    inputs.push(nextId)
    return nextId
}

console.log('[Initializing Model]')
Util.load_encoder_hparams_and_params(Util.config.model_size, Util.config.models_dir).then(({encoder, hparams, params}) => {
    let shouldIterate = true
    while (shouldIterate) {
        const prompt = question("?>")
        const input_ids = encoder.encode(prompt)
        if (prompt.toLowerCase() === 'q') {
            shouldIterate = false
            break
        }
    
        if (input_ids.length + Util.config.n_tokens_to_generate >= hparams['n_ctx']) {
            console.error(`Error: hparams[n_ctx] and input_ids length mismatch, n_ctx=${hparams['n_ctx']}`)
            break
        }
    
        for (let i = 0; i < Util.config.n_tokens_to_generate; i++) {
            generateToken(input_ids, params, hparams['n_head'], 1)
        }
        const output_text = encoder.decode(output_ids)
        console.log(output_text)
    
    }
})