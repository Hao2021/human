const { GoogleGenAI } = require("@google/genai");
const { runCausalSimulation } = require("./engine");

// IMPORTANT: This key is loaded securely from the hosting platform's environment variables (GEMINI_API_KEY).
const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
    // If the key is not set, this error will be returned to the client, preventing exposure.
    throw new Error("GEMINI_API_KEY environment variable not set.");
}
const ai = new GoogleGenAI({ apiKey });
const MODEL_NAME = 'gemini-2.5-flash-preview-09-2025';

// --- JSON Schemas ---

// Schema for the 'simulate' action
const CAUSAL_GRAPH_SCHEMA = {
    type: "OBJECT",
    properties: {
        "variables": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": { "type": "STRING" },
                    "value": { "type": "NUMBER" },
                    "baseline": { "type": "NUMBER" }
                },
                "required": ["id"]
            }
        },
        "edges": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "from": { "type": "STRING" },
                    "to": { "type": "STRING" },
                    "weight": { "type": "NUMBER" }
                },
                "required": ["from", "to", "weight"]
            }
        }
    }
};

const SIMULATION_SCHEMA = {
    type: "OBJECT",
    properties: {
        "analysis": {
            "type": "OBJECT",
            "properties": {
                "Geneticist": { "type": "STRING" },
                "Neuroscientist": { "type": "STRING" },
                "Physiologist": { "type": "STRING" },
                "Neurochemist": { "type": "STRING" },
                "Psychoneuroimmunologist": { "type": "STRING" },
                "CognitivePsychologist": { "type": "STRING" },
                "DevelopmentalPsychologist": { "type": "STRING" },
                "Educator": { "type": "STRING" },
                "Sociologist": { "type": "STRING" },
                "EconomistEntrepreneur": { "type": "STRING" },
                "Philosopher": { "type": "STRING" }
            },
            "required": ["Geneticist", "Neuroscientist", "Physiologist", "Neurochemist", "Psychoneuroimmunologist", "CognitivePsychologist", "DevelopmentalPsychologist", "Educator", "Sociologist", "EconomistEntrepreneur", "Philosopher"]
        },
        "conclusionSynthesis": { "type": "STRING" },
        "newState": {
            "type": "OBJECT",
            "properties": {
                "vitality": { "type": "NUMBER" },
                "cognition": { "type": "NUMBER" },
                "emotion": { "type": "NUMBER" },
                "adaptability": { "type": "NUMBER" },
                "meaning": { "type": "NUMBER" }
            },
            "required": ["vitality", "cognition", "emotion", "adaptability", "meaning"]
        },
        "causalGraph": CAUSAL_GRAPH_SCHEMA,
        "attractor": { "type": "STRING" },
        "leverageUsed": { "type": "STRING" },
        "interventionsPlanned": {
            "type": "ARRAY",
            "items": { "type": "STRING" }
        }
    },
    "required": ["analysis", "conclusionSynthesis", "newState"]
};

// Schema for the 'recommend' action
const RECOMMENDATION_SCHEMA = {
    type: "OBJECT",
    properties: {
        "recommendation": { "type": "STRING" },
        "newState": {
            "type": "OBJECT",
            "properties": {
                "vitality": { "type": "NUMBER" },
                "cognition": { "type": "NUMBER" },
                "emotion": { "type": "NUMBER" },
                "adaptability": { "type": "NUMBER" },
                "meaning": { "type": "NUMBER" }
            },
            "required": ["vitality", "cognition", "emotion", "adaptability", "meaning"]
        },
        "causalGraph": CAUSAL_GRAPH_SCHEMA,
        "attractor": { "type": "STRING" },
        "leverageUsed": { "type": "STRING" },
        "interventionsPlanned": {
            "type": "ARRAY",
            "items": { "type": "STRING" }
        }
    },
    "required": ["recommendation", "newState"]
};

// Define the system prompt base used for both types of calls
const SYSTEM_PROMPT_BASE = `You are running a research simulation on human potential as a System of Systems.
The system has 11 expert agents (Geneticist, Neuroscientist, Performance Physiologist, Neurochemist, Psychoneuroimmunologist, Cognitive Psychologist, Developmental Psychologist, Educator, Sociologist, Economist/Entrepreneur, Philosopher).
The human system state is defined by 5 factors (Vitality, Cognition, Emotion, Adaptability, Meaning) rated 1-10.`;

const WELLNESS_METRIC_LABELS = {
    hrvMs: 'Heart Rate Variability (ms)',
    sleepDurationHrs: 'Total Sleep Duration (hrs)',
    deepSleepHrs: 'Deep Sleep (hrs)',
    remSleepHrs: 'REM Sleep (hrs)',
    lightSleepHrs: 'Light Sleep (hrs)',
    restingHeartRateBpm: 'Resting Heart Rate (bpm)'
};

const sanitizeWellnessMetrics = (metrics) => {
    if (!metrics || typeof metrics !== 'object') {
        return null;
    }

    const sanitized = {};
    let hasValues = false;

    Object.keys(WELLNESS_METRIC_LABELS).forEach((key) => {
        const value = Number(metrics[key]);
        if (Number.isFinite(value)) {
            sanitized[key] = value;
            hasValues = true;
        }
    });

    return hasValues ? sanitized : null;
};

const formatWellnessMetricsForPrompt = (metrics) => {
    if (!metrics) {
        return '';
    }

    return Object.entries(metrics)
        .map(([key, value]) => {
            const label = WELLNESS_METRIC_LABELS[key] || key;
            return `${label}: ${value}`;
        })
        .join(', ');
};

const deepClone = (value) => JSON.parse(JSON.stringify(value));

const DEFAULT_CAUSAL_GRAPH = {
    variables: [
        { id: 'Isolation', value: 6, baseline: 5 },
        { id: 'Belonging', value: 4, baseline: 5 },
        { id: 'HPA', value: 6, baseline: 5 },
        { id: 'Inflammation', value: 5.5, baseline: 5 },
        { id: 'Neurodegeneration', value: 5.2, baseline: 5 },
        { id: 'Withdrawal', value: 5.8, baseline: 5 },
        { id: 'VagalTone', value: 4.8, baseline: 5 },
        { id: 'Meaning', value: 4.5, baseline: 5 }
    ],
    edges: [
        { from: 'Isolation', to: 'HPA', weight: 0.65 },
        { from: 'HPA', to: 'Inflammation', weight: 0.7 },
        { from: 'Inflammation', to: 'Neurodegeneration', weight: 0.55 },
        { from: 'Neurodegeneration', to: 'Withdrawal', weight: 0.42 },
        { from: 'Withdrawal', to: 'Isolation', weight: 0.58 },
        { from: 'Belonging', to: 'Isolation', weight: -0.62 },
        { from: 'Belonging', to: 'VagalTone', weight: 0.57 },
        { from: 'VagalTone', to: 'HPA', weight: -0.68 },
        { from: 'Meaning', to: 'Belonging', weight: 0.6 },
        { from: 'Meaning', to: 'Isolation', weight: -0.45 },
        { from: 'Isolation', to: 'Meaning', weight: -0.35 },
        { from: 'VagalTone', to: 'Meaning', weight: 0.38 }
    ]
};

const buildDefaultCausalGraph = () => deepClone(DEFAULT_CAUSAL_GRAPH);

const ensureCausalGraph = (graph) => {
    if (!graph || typeof graph !== 'object') {
        return buildDefaultCausalGraph();
    }

    const resolved = buildDefaultCausalGraph();

    if (Array.isArray(graph.variables)) {
        resolved.variables = deepClone(graph.variables);
    } else if (Array.isArray(graph.nodes)) {
        resolved.variables = deepClone(graph.nodes);
    } else if (Array.isArray(graph.stocks)) {
        resolved.variables = deepClone(graph.stocks);
    }

    if (Array.isArray(graph.edges)) {
        resolved.edges = deepClone(graph.edges);
    } else if (Array.isArray(graph.links)) {
        resolved.edges = deepClone(graph.links);
    } else if (Array.isArray(graph.connections)) {
        resolved.edges = deepClone(graph.connections);
    }

    return resolved;
};

const coerceString = (value, fallback = '') => {
    if (typeof value === 'string') {
        const trimmed = value.trim();
        return trimmed.length ? trimmed : fallback;
    }
    return fallback;
};

const ensureInterventionsArray = (value) => {
    if (Array.isArray(value)) {
        return value
            .map((entry) => (typeof entry === 'string' ? entry.trim() : ''))
            .filter((entry) => entry.length > 0);
    }

    if (typeof value === 'string') {
        const trimmed = value.trim();
        return trimmed ? [trimmed] : [];
    }

    return [];
};

const safeNumber = (value) => {
    const num = Number(value);
    return Number.isFinite(num) ? num : 0;
};

const computeAttractor = (values = {}) => {
    const threat = safeNumber(values.HPA) + safeNumber(values.Inflammation) + safeNumber(values.Isolation);
    const flow = safeNumber(values.Belonging) + safeNumber(values.VagalTone) + safeNumber(values.Meaning);

    if (threat - flow > 1) {
        return 'Threat-Survival';
    }
    if (flow - threat > 1) {
        return 'Eudaemonic-Flow';
    }
    return 'Mixed/Transition';
};

// --- Serverless Function Handler ---
module.exports = async (req, res) => {
    // Vercel/Netlify auto-parses the body for POST requests
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed' });
    }

    try {
        const { eventText, currentState, type, wellnessMetrics } = req.body;
        const sanitizedWellnessMetrics = sanitizeWellnessMetrics(wellnessMetrics);

        let userQuery, responseSchema, systemPrompt;

        if (type === 'simulate') {
            systemPrompt = `${SYSTEM_PROMPT_BASE}
            Your task is to:
            1. Receive a user's descriptive text about a life event.
            2. Simulate the immediate impact of this event across ALL 11 expert subsystems, providing a concise, 1-sentence analysis for each.
            3. Provide a single "Conclusion & Synthesis" summarizing the emergent system behavior.
            4. Generate a *new, plausible state vector* (1-10) that results from the event and systemic adaptation.`;

            const wellnessSummary = formatWellnessMetricsForPrompt(sanitizedWellnessMetrics);
            const metricsLine = wellnessSummary ? `Sleep & Recovery Metrics: ${wellnessSummary}\n` : '';

            userQuery = `${metricsLine}Current State: ${JSON.stringify(currentState)}
            Life Event to Simulate: "${eventText}"
            Simulate the full system response to this event.`;
            responseSchema = SIMULATION_SCHEMA;

        } else if (type === 'recommend') {
            systemPrompt = `${SYSTEM_PROMPT_BASE}
            Your task is to:
            1. Analyze the current state vector.
            2. Provide a 3-4 sentence "Recommendation" of a practical, systemic action.
            3. Generate a *new, "improved" state vector* (1-10) that would *realistically result* from following your recommendation.`;
            
            userQuery = `Current State: ${JSON.stringify(currentState)}
            Analyze this state and provide a recommendation and the resulting new state vector.`;
            responseSchema = RECOMMENDATION_SCHEMA;
        } else {
            return res.status(400).json({ error: 'Invalid API call type. Must be "simulate" or "recommend".' });
        }
        
        const response = await ai.models.generateContent({
            model: MODEL_NAME,
            contents: [{
                role: "user",
                parts: [{ text: userQuery }]
            }],
            config: {
                systemInstruction: systemPrompt,
                responseMimeType: "application/json",
                responseSchema: responseSchema,
                temperature: 0.8
            }
        });

        // The model returns the JSON as a string in the text field
        const jsonText = response.text.replace(/```json|```/g, '').trim();
        const parsedResponse = JSON.parse(jsonText);

        const finalResponse = { ...parsedResponse };

        if (type === 'simulate') {
            const resolvedCausalGraph = ensureCausalGraph(parsedResponse.causalGraph);
            const simulationOutput = runCausalSimulation(resolvedCausalGraph, {
                initialState: currentState,
                steps: 16,
                dt: 0.32,
                damping: 0.26
            });

            const timeSeries = Array.isArray(simulationOutput.timeSeries) ? simulationOutput.timeSeries : [];
            const latestValues = timeSeries.length ? timeSeries[timeSeries.length - 1].values : {};

            finalResponse.causalGraph = resolvedCausalGraph;
            finalResponse.timeSeries = timeSeries;
            finalResponse.loopsDetectedStructured = simulationOutput.loopsDetected || [];
            finalResponse.loopsDetected = simulationOutput.loopsDetected || [];
            finalResponse.newState = simulationOutput.newState;
            finalResponse.attractor = parsedResponse.attractor || computeAttractor(latestValues);
            finalResponse.leverageUsed = coerceString(parsedResponse.leverageUsed, 'Not specified');
            finalResponse.interventionsPlanned = ensureInterventionsArray(parsedResponse.interventionsPlanned);

        } else if (type === 'recommend') {
            finalResponse.leverageUsed = coerceString(parsedResponse.leverageUsed, 'Not specified');
            finalResponse.interventionsPlanned = ensureInterventionsArray(parsedResponse.interventionsPlanned);
            if (parsedResponse.causalGraph) {
                finalResponse.causalGraph = ensureCausalGraph(parsedResponse.causalGraph);
            }
        }

        // Return the enriched JSON back to the client
        res.status(200).json(finalResponse);

    } catch (error) {
        console.error('Serverless Function Error:', error);
        // Send a generic 500 error to the client
        res.status(500).json({ 
            error: 'Internal Server Error during AI processing.', 
            message: error.message 
        });
    }
};


// const { GoogleGenAI } = require("@google/genai");

// // IMPORTANT: This key is loaded securely from the hosting platform's environment variables (GEMINI_API_KEY).
// const apiKey = process.env.GEMINI_API_KEY;
// if (!apiKey) {
//     // If the key is not set, this error will be returned to the client, preventing exposure.
//     throw new Error("GEMINI_API_KEY environment variable not set.");
// }
// const ai = new GoogleGenAI({ apiKey });
// const MODEL_NAME = 'gemini-2.5-flash-preview-09-2025';

// // --- JSON Schemas ---

// // Schema for the 'simulate' action
// const SIMULATION_SCHEMA = {
//     type: "OBJECT",
//     properties: {
//         "analysis": {
//             "type": "OBJECT",
//             "properties": {
//                 "Geneticist": { "type": "STRING" },
//                 "Neuroscientist": { "type": "STRING" },
//                 "Physiologist": { "type": "STRING" },
//                 "Neurochemist": { "type": "STRING" },
//                 "Psychoneuroimmunologist": { "type": "STRING" },
//                 "CognitivePsychologist": { "type": "STRING" },
//                 "DevelopmentalPsychologist": { "type": "STRING" },
//                 "Educator": { "type": "STRING" },
//                 "Sociologist": { "type": "STRING" },
//                 "EconomistEntrepreneur": { "type": "STRING" },
//                 "Philosopher": { "type": "STRING" }
//             },
//             "required": ["Geneticist", "Neuroscientist", "Physiologist", "Neurochemist", "Psychoneuroimmunologist", "CognitivePsychologist", "DevelopmentalPsychologist", "Educator", "Sociologist", "EconomistEntrepreneur", "Philosopher"]
//         },
//         "conclusionSynthesis": { "type": "STRING" },
//         "newState": {
//             "type": "OBJECT",
//             "properties": {
//                 "vitality": { "type": "NUMBER" },
//                 "cognition": { "type": "NUMBER" },
//                 "emotion": { "type": "NUMBER" },
//                 "adaptability": { "type": "NUMBER" },
//                 "meaning": { "type": "NUMBER" }
//             },
//             "required": ["vitality", "cognition", "emotion", "adaptability", "meaning"]
//         }
//     },
//     "required": ["analysis", "conclusionSynthesis", "newState"]
// };

// // Schema for the 'recommend' action
// const RECOMMENDATION_SCHEMA = {
//     type: "OBJECT",
//     properties: {
//         "recommendation": { "type": "STRING" },
//         "newState": {
//             "type": "OBJECT",
//             "properties": {
//                 "vitality": { "type": "NUMBER" },
//                 "cognition": { "type": "NUMBER" },
//                 "emotion": { "type": "NUMBER" },
//                 "adaptability": { "type": "NUMBER" },
//                 "meaning": { "type": "NUMBER" }
//             },
//             "required": ["vitality", "cognition", "emotion", "adaptability", "meaning"]
//         }
//     },
//     "required": ["recommendation", "newState"]
// };

// // Define the system prompt base used for both types of calls
// const SYSTEM_PROMPT_BASE = `You are running a research simulation on human potential as a System of Systems.
// The system has 11 expert agents (Geneticist, Neuroscientist, Performance Physiologist, Neurochemist, Psychoneuroimmunologist, Cognitive Psychologist, Developmental Psychologist, Educator, Sociologist, Economist/Entrepreneur, Philosopher).
// The human system state is defined by 5 factors (Vitality, Cognition, Emotion, Adaptability, Meaning) rated 1-10.`;

// // --- Serverless Function Handler ---
// module.exports = async (req, res) => {
//     // Vercel/Netlify auto-parses the body for POST requests
//     if (req.method !== 'POST') {
//         return res.status(405).json({ error: 'Method Not Allowed' });
//     }

//     try {
//         // We ensure the body is available (it should be parsed by the runtime)
//         const { eventText, currentState, type } = req.body;
        
//         let userQuery, responseSchema, systemPrompt;

//         if (type === 'simulate') {
//             systemPrompt = `${SYSTEM_PROMPT_BASE}
//             Your task is to:
//             1. Receive a user's descriptive text about a life event.
//             2. Simulate the immediate impact of this event across ALL 11 expert subsystems, providing a concise, 1-sentence analysis for each.
//             3. Provide a single "Conclusion & Synthesis" summarizing the emergent system behavior.
//             4. Generate a *new, plausible state vector* (1-10) that results from the event and systemic adaptation.`;
            
//             userQuery = `Current State: ${JSON.stringify(currentState)}
//             Life Event to Simulate: "${eventText}"
//             Simulate the full system response to this event.`;
//             responseSchema = SIMULATION_SCHEMA;

//         } else if (type === 'recommend') {
//             systemPrompt = `${SYSTEM_PROMPT_BASE}
//             Your task is to:
//             1. Analyze the current state vector.
//             2. Provide a 3-4 sentence "Recommendation" of a practical, systemic action.
//             3. Generate a *new, "improved" state vector* (1-10) that would *realistically result* from following your recommendation.`;
            
//             userQuery = `Current State: ${JSON.stringify(currentState)}
//             Analyze this state and provide a recommendation and the resulting new state vector.`;
//             responseSchema = RECOMMENDATION_SCHEMA;
//         } else {
//             return res.status(400).json({ error: 'Invalid API call type. Must be "simulate" or "recommend".' });
//         }
        
//         const response = await ai.models.generateContent({
//             model: MODEL_NAME,
//             contents: [{
//                 role: "user",
//                 parts: [{ text: userQuery }]
//             }],
//             config: {
//                 systemInstruction: systemPrompt,
//                 responseMimeType: "application/json",
//                 responseSchema: responseSchema,
//                 temperature: 0.8
//             }
//         });

//         // The model returns the JSON as a string in the text field
//         const jsonText = response.text.replace(/```json|```/g, '').trim();
//         const parsedResponse = JSON.parse(jsonText);
        
//         // Return the parsed JSON back to the client
//         res.status(200).json(parsedResponse);

//     } catch (error) {
//         console.error('Serverless Function Error:', error);
//         // Send a generic 500 error to the client
//         res.status(500).json({ 
//             error: 'Internal Server Error during AI processing.', 
//             message: error.message 
//         });
//     }
// };
// ```eof