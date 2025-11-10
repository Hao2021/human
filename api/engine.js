"use strict";

const clamp = (value, min, max) => {
    if (Number.isNaN(value)) {
        return min;
    }
    return Math.max(min, Math.min(max, value));
};

const toNumber = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
};

const NORMALIZED_RANGE = { min: 0, max: 10 };

const normalizeVariables = (graph = {}) => {
    const rawNodes = graph.variables || graph.nodes || graph.stocks || graph.vertices || graph;
    const normalized = [];

    if (Array.isArray(rawNodes)) {
        rawNodes.forEach((item) => {
            if (!item) return;
            if (typeof item === "string") {
                normalized.push({ id: item, value: 5, baseline: 5 });
                return;
            }
            const id = item.id || item.name;
            if (!id) return;
            const initial = toNumber(item.initial ?? item.value ?? item.start ?? 5, 5);
            const baseline = toNumber(item.baseline ?? item.setPoint ?? initial ?? 5, 5);
            normalized.push({
                id,
                value: clamp(initial, NORMALIZED_RANGE.min, NORMALIZED_RANGE.max),
                baseline: clamp(baseline, NORMALIZED_RANGE.min, NORMALIZED_RANGE.max)
            });
        });
        return normalized;
    }

    if (rawNodes && typeof rawNodes === "object") {
        Object.entries(rawNodes).forEach(([id, item]) => {
            if (!id) return;
            if (item && typeof item === "object") {
                const initial = toNumber(item.initial ?? item.value ?? item.start ?? 5, 5);
                const baseline = toNumber(item.baseline ?? item.setPoint ?? initial ?? 5, 5);
                normalized.push({
                    id,
                    value: clamp(initial, NORMALIZED_RANGE.min, NORMALIZED_RANGE.max),
                    baseline: clamp(baseline, NORMALIZED_RANGE.min, NORMALIZED_RANGE.max)
                });
            } else {
                const initial = toNumber(item, 5);
                normalized.push({ id, value: clamp(initial, NORMALIZED_RANGE.min, NORMALIZED_RANGE.max), baseline: clamp(initial, NORMALIZED_RANGE.min, NORMALIZED_RANGE.max) });
            }
        });
        return normalized;
    }

    return [];
};

const normalizeEdges = (graph = {}) => {
    const rawEdges = graph.edges || graph.links || graph.connections || [];
    const normalized = [];

    if (Array.isArray(rawEdges)) {
        rawEdges.forEach((edge) => {
            if (!edge) return;
            const from = edge.from || edge.source;
            const to = edge.to || edge.target;
            if (!from || !to) return;
            const weight = toNumber(edge.weight ?? edge.strength ?? edge.value ?? 0, 0);
            normalized.push({ from, to, weight });
        });
    } else if (rawEdges && typeof rawEdges === "object") {
        Object.entries(rawEdges).forEach(([key, value]) => {
            if (!value) return;
            if (Array.isArray(value)) {
                value.forEach((target) => {
                    if (!target) return;
                    const to = typeof target === "string" ? target : target.to || target.target;
                    const weight = typeof target === "object" ? toNumber(target.weight ?? target.strength ?? target.value ?? 0, 0) : 0.5;
                    normalized.push({ from: key, to, weight });
                });
            } else if (typeof value === "object") {
                const from = value.from || value.source || key;
                const to = value.to || value.target;
                if (!to) return;
                const weight = toNumber(value.weight ?? value.strength ?? value.value ?? 0, 0);
                normalized.push({ from, to, weight });
            }
        });
    }

    return normalized;
};

const buildAdjacency = (edges) => {
    const adjacency = new Map();
    edges.forEach((edge) => {
        if (!adjacency.has(edge.from)) {
            adjacency.set(edge.from, []);
        }
        adjacency.get(edge.from).push({ to: edge.to, weight: edge.weight });
    });
    return adjacency;
};

const canonicalizeCycle = (nodes) => {
    if (!nodes.length) return null;
    const simple = nodes.slice(0, -1);
    const n = simple.length;
    if (n === 0) return null;
    const rotations = [];
    for (let i = 0; i < n; i += 1) {
        const rotated = simple.slice(i).concat(simple.slice(0, i));
        rotations.push(rotated);
    }
    rotations.sort((a, b) => {
        for (let i = 0; i < a.length; i += 1) {
            if (a[i] === b[i]) continue;
            return a[i] < b[i] ? -1 : 1;
        }
        return 0;
    });
    const canonical = rotations[0];
    return canonical.concat(canonical[0]).join("->");
};

const detectCycles = (edges) => {
    if (!edges.length) {
        return [];
    }
    const adjacency = buildAdjacency(edges);
    const nodes = Array.from(new Set(edges.flatMap((edge) => [edge.from, edge.to])));
    const maxAbs = edges.reduce((acc, edge) => Math.max(acc, Math.abs(edge.weight)), 0) || 1;
    const seen = new Map();

    const stack = [];
    const edgeStack = [];

    const visit = (current, start) => {
        stack.push(current);
        const outgoing = adjacency.get(current) || [];
        for (const edge of outgoing) {
            edgeStack.push(edge.weight);
            const idx = stack.indexOf(edge.to);
            if (idx !== -1) {
                if (stack.length - idx >= 1) {
                    const cycleNodes = stack.slice(idx).concat(edge.to);
                    const cycleWeights = edgeStack.slice(idx);
                    const key = canonicalizeCycle(cycleNodes);
                    if (key && !seen.has(key)) {
                        const product = cycleWeights.reduce((acc, weight) => acc * (weight === 0 ? 0 : Math.sign(weight)), 1);
                        const avgAbs = cycleWeights.reduce((acc, weight) => acc + Math.abs(weight), 0) / cycleWeights.length;
                        const dominance = Math.round(clamp((avgAbs / maxAbs) * 100, 0, 100));
                        const type = product > 0 ? "Reinforcing" : "Balancing";
                        const chain = cycleNodes.join(" → ");
                        seen.set(key, {
                            type,
                            dominance,
                            nodes: cycleNodes.slice(0, -1),
                            chain,
                            weights: cycleWeights
                        });
                    }
                }
            } else {
                visit(edge.to, start);
            }
            edgeStack.pop();
        }
        stack.pop();
    };

    nodes.forEach((node) => {
        visit(node, node);
        stack.length = 0;
        edgeStack.length = 0;
    });

    return Array.from(seen.values());
};

const deriveFiveFactorState = (finalValues) => {
    const getValue = (name) => {
        if (finalValues.has(name)) {
            return finalValues.get(name);
        }
        return 5;
    };

    const aggregate = (positives, negatives) => {
        const posValues = positives.length ? positives.map((name) => getValue(name)) : [5];
        const negValues = negatives.length ? negatives.map((name) => getValue(name)) : [5];
        const posAvg = posValues.reduce((sum, value) => sum + value, 0) / posValues.length;
        const negAvg = negValues.reduce((sum, value) => sum + value, 0) / negValues.length;
        const adjusted = 5 + (posAvg - 5) * 0.6 - (negAvg - 5) * 0.6;
        return clamp(adjusted, 1, 10);
    };

    const state = {
        vitality: aggregate(["Belonging", "VagalTone"], ["Isolation", "HPA", "Inflammation", "Neurodegeneration"]),
        cognition: aggregate(["Meaning", "Belonging"], ["Isolation", "Neurodegeneration", "Withdrawal"]),
        emotion: aggregate(["Belonging", "Meaning", "VagalTone"], ["Isolation", "Withdrawal", "HPA"]),
        adaptability: aggregate(["VagalTone", "Meaning", "Belonging"], ["HPA", "Inflammation", "Isolation"]),
        meaning: aggregate(["Meaning", "Belonging"], ["Isolation", "Withdrawal"])
    };

    return state;
};

const runEulerSimulation = (variables, edges, opts = {}) => {
    const steps = Number.isInteger(opts.steps) && opts.steps > 0 ? opts.steps : 12;
    const dt = typeof opts.dt === "number" && Number.isFinite(opts.dt) ? opts.dt : 0.35;
    const damping = typeof opts.damping === "number" && Number.isFinite(opts.damping) ? opts.damping : 0.28;

    const values = new Map();
    const baselines = new Map();

    variables.forEach(({ id, value, baseline }) => {
        values.set(id, clamp(value, NORMALIZED_RANGE.min, NORMALIZED_RANGE.max));
        baselines.set(id, clamp(baseline, NORMALIZED_RANGE.min, NORMALIZED_RANGE.max));
    });

    const timeSeries = [];

    const captureSnapshot = (step) => {
        const snapshot = {};
        values.forEach((value, key) => {
            snapshot[key] = value;
        });
        timeSeries.push({ step, values: snapshot });
    };

    captureSnapshot(0);

    for (let step = 1; step <= steps; step += 1) {
        const deltas = new Map();
        values.forEach((_, key) => {
            deltas.set(key, 0);
        });

        edges.forEach(({ from, to, weight }) => {
            if (!values.has(from) || !values.has(to)) {
                return;
            }
            const influence = values.get(from) * weight;
            deltas.set(to, (deltas.get(to) || 0) + influence);
        });

        values.forEach((currentValue, key) => {
            const baseline = baselines.get(key) ?? 5;
            const delta = deltas.get(key) ?? 0;
            const nextValue = currentValue + dt * (delta - damping * (currentValue - baseline));
            values.set(key, clamp(nextValue, NORMALIZED_RANGE.min, NORMALIZED_RANGE.max));
        });

        captureSnapshot(step);
    }

    return { timeSeries, finalValues: values };
};

const VARIABLE_FACTORS = {
    Isolation: ["emotion", "meaning"],
    Belonging: ["emotion", "meaning"],
    HPA: ["vitality", "adaptability"],
    Inflammation: ["vitality", "adaptability"],
    Neurodegeneration: ["cognition", "vitality"],
    Withdrawal: ["emotion", "adaptability"],
    VagalTone: ["vitality", "adaptability", "emotion"],
    Meaning: ["meaning", "cognition"]
};

const applyInitialStateAnchors = (variables, initialState) => {
    if (!initialState || typeof initialState !== "object") {
        return variables;
    }

    return variables.map((variable) => {
        const anchors = VARIABLE_FACTORS[variable.id];
        if (!anchors || !anchors.length) {
            return variable;
        }
        const anchorValues = anchors
            .map((factor) => toNumber(initialState[factor], Number.NaN))
            .filter((value) => Number.isFinite(value));

        if (!anchorValues.length) {
            return variable;
        }

        const average = anchorValues.reduce((sum, value) => sum + value, 0) / anchorValues.length;
        const clamped = clamp(average, NORMALIZED_RANGE.min, NORMALIZED_RANGE.max);
        return {
            ...variable,
            value: clamped,
            baseline: clamped
        };
    });
};

const runCausalSimulation = (causalGraph = {}, opts = {}) => {
    const variables = normalizeVariables(causalGraph);
    const edges = normalizeEdges(causalGraph);

    const normalizedVariables = variables.length ? variables : [
        { id: "Isolation", value: 6, baseline: 5 },
        { id: "Belonging", value: 4, baseline: 5 },
        { id: "HPA", value: 6, baseline: 5 },
        { id: "Inflammation", value: 5.5, baseline: 5 },
        { id: "Neurodegeneration", value: 5.2, baseline: 5 },
        { id: "Withdrawal", value: 5.8, baseline: 5 },
        { id: "VagalTone", value: 4.8, baseline: 5 },
        { id: "Meaning", value: 4.5, baseline: 5 }
    ];

    const normalizedEdges = edges.length ? edges : [
        { from: "Isolation", to: "HPA", weight: 0.65 },
        { from: "HPA", to: "Inflammation", weight: 0.7 },
        { from: "Inflammation", to: "Neurodegeneration", weight: 0.55 },
        { from: "Neurodegeneration", to: "Withdrawal", weight: 0.42 },
        { from: "Withdrawal", to: "Isolation", weight: 0.58 },
        { from: "Belonging", to: "Isolation", weight: -0.62 },
        { from: "Belonging", to: "VagalTone", weight: 0.57 },
        { from: "VagalTone", to: "HPA", weight: -0.68 },
        { from: "Meaning", to: "Belonging", weight: 0.6 },
        { from: "Meaning", to: "Isolation", weight: -0.45 },
        { from: "Isolation", to: "Meaning", weight: -0.35 },
        { from: "VagalTone", to: "Meaning", weight: 0.38 }
    ];

    const anchoredVariables = applyInitialStateAnchors(normalizedVariables, opts.initialState);
    const { timeSeries, finalValues } = runEulerSimulation(anchoredVariables, normalizedEdges, opts);
    const loopsDetected = detectCycles(normalizedEdges);
    const newState = deriveFiveFactorState(finalValues);

    return {
        loopsDetected,
        timeSeries,
        newState
    };
};

module.exports = {
    runCausalSimulation
};
