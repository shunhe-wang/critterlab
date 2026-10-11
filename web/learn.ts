import { probabilities, type Policy } from './engine';
export interface Lesson { observation: number[]; action: number }

/** Actual cross-entropy backpropagation through the exported dense/tanh actor.
 * This is an exploratory, session-local teaching toy; published offline results
 * are independent of these visitor-supplied demonstrations.
 */
export async function teach(source: Policy, lessons: Lesson[], epochs = 35): Promise<Policy> {
  if (!lessons.length || lessons.length > 2400) throw new Error('Teaching needs 1–2400 demonstrated actions');
  const policy: Policy = structuredClone(source);
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (const lesson of lessons) {
      const activations: number[][] = [lesson.observation];
      policy.layers.forEach((layer, index) => {
        const input = activations.at(-1)!;
        const output = layer.weight.map((row, i) => row.reduce((sum, w, j) => sum + w * input[j], layer.bias[i]));
        activations.push(index === policy.layers.length - 1 ? output : output.map(Math.tanh));
      });
      let delta = probabilities(policy, lesson.observation);
      delta[lesson.action] -= 1;
      for (let layerIndex = policy.layers.length - 1; layerIndex >= 0; layerIndex--) {
        const layer = policy.layers[layerIndex], input = activations[layerIndex];
        const upstream = input.map((_, j) => layer.weight.reduce((sum, row, i) => sum + row[j] * delta[i], 0));
        layer.weight.forEach((row, i) => {
          row.forEach((_, j) => { row[j] -= .008 * delta[i] * input[j]; });
          layer.bias[i] -= .008 * delta[i];
        });
        delta = upstream.map((v, j) => v * (1 - input[j] ** 2));
      }
    }
    if (epoch % 5 === 0) await new Promise<void>(resolve => setTimeout(resolve, 0));
  }
  policy.label = 'Your lessons';
  return policy;
}
