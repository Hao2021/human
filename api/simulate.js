const { GoogleGenAI } = require("@google/genai");

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
        }
    },
    "required": ["recommendation", "newState"]
};

// Define the system prompt base used for both types of calls
const SYSTEM_PROMPT_BASE = `You are running a research simulation on human potential as a System of Systems.
The system has 11 expert agents (Geneticist, Neuroscientist, Performance Physiologist, Neurochemist, Psychoneuroimmunologist, Cognitive Psychologist, Developmental Psychologist, Educator, Sociologist, Economist/Entrepreneur, Philosopher).
The human system state is defined by 5 factors (Vitality, Cognition, Emotion, Adaptability, Meaning) rated 1-10.`;

// --- Serverless Function Handler ---
module.exports = async (req, res) => {
    // Vercel/Netlify auto-parses the body for POST requests
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed' });
    }

    try {
        const { eventText, currentState, type } = req.body;
        
        let userQuery, responseSchema, systemPrompt;

        if (type === 'simulate') {
            systemPrompt = `${SYSTEM_PROMPT_BASE}
            Your task is to:
            1. Receive a user's descriptive text about a life event.
            2. Simulate the immediate impact of this event across ALL 11 expert subsystems, providing a concise, 1-sentence analysis for each.
            3. Provide a single "Conclusion & Synthesis" summarizing the emergent system behavior.
            4. Generate a *new, plausible state vector* (1-10) that results from the event and systemic adaptation.`;
            
            userQuery = `Current State: ${JSON.stringify(currentState)}
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
        
        // Return the parsed JSON back to the client
        res.status(200).json(parsedResponse);

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