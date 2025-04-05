import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { Document } from "langchain/document";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { WebPDFLoader } from "@langchain/community/document_loaders/web/pdf";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";


interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    { id: Date.now(), text: 'Hello! I am a RAG-powered chatbot. You can ask me questions about the documents you upload.', sender: 'bot' }
  ]);
  const [input, setInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [vectorStore, setVectorStore] = useState<MemoryVectorStore | null>(null);
  const [apiKey, setApiKey] = useState<string>('');
  const [isApiKeySet, setIsApiKeySet] = useState<boolean>(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleApiKeySubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (apiKey.trim() !== '') {
      setIsApiKeySet(true);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || !isApiKeySet) return;
    
    setIsLoading(true);
    
    try {
      const file = e.target.files[0];
      
      // Add a message about the file being processed
      addMessage(`Processing document: ${file.name}`, 'bot');
      
      // Detect if the file is a PDF (checking both MIME type and extension)
      const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
      
      // Create text splitter (used for both file types)
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1250,
        chunkOverlap: 150
      });
      
      let docs;
      
      if (isPdf) {
        // Handle PDF file
        const loader = new WebPDFLoader(file);
        const pdfDocs = await loader.load();
        
        // Split PDF documents
        docs = await splitter.splitDocuments(pdfDocs);
      } else {
        // Handle text file (as before)
        const text = await file.text();
        
        // Split text into chunks
        const chunks = await splitter.splitText(text);
        
        // Create documents from chunks
        docs = chunks.map(chunk => new Document({ pageContent: chunk }));
      }
      
      // Create vector store
      const embeddings = new OpenAIEmbeddings({ openAIApiKey: apiKey });
      const store = await MemoryVectorStore.fromDocuments(docs, embeddings);
      setVectorStore(store);
      
      addMessage(`Document processed successfully! You can now ask questions about ${file.name}.`, 'bot');
    } catch (error) {
      console.error("Error processing document:", error);
      addMessage("There was an error processing your document. Please try again.", 'bot');
    } finally {
      setIsLoading(false);
      // Reset file input
      if (e.target) e.target.value = '';
    }
  };

  const addMessage = (text: string, sender: 'user' | 'bot') => {
    const newMessage: Message = {
      id: Date.now() + Math.random(), // Ensure unique IDs even for messages created at the same time
      text,
      sender
    };
    setMessages(prevMessages => [...prevMessages, newMessage]);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (input.trim() === '' || !isApiKeySet) return;
    
    // Add user message
    const userMessage = input.trim();
    addMessage(userMessage, 'user');
    setInput('');
    
    setIsLoading(true);
    
    try {
      if (vectorStore) {
        // Use RAG to answer the question
        const model = new ChatOpenAI({ 
          openAIApiKey: apiKey, 
          temperature: 0.7,
          modelName: "gpt-3.5-turbo"
        });
        
        const retriever = vectorStore.asRetriever();
        
        const prompt = ChatPromptTemplate.fromTemplate(`
          You are a helpful assistant that answers questions about academic documents.
          
          Use the following context to answer the question. If the answer is not in the context, 
          say "I don't have enough information about that in the document."
          
          Context:
          {context}
          
          Question: {input}
          
          Answer:
        `);
        
        const documentChain = await createStuffDocumentsChain({
          llm: model,
          prompt,
        });
        
        const retrievalChain = await createRetrievalChain({
          combineDocsChain: documentChain,
          retriever,
        });
        
        const response = await retrievalChain.invoke({
          input: userMessage,
        });
        
        // Handle different response formats
// Modified code for the response handling section:
let botResponse = "I couldn't find a relevant answer in the document.";

if ('answer' in response && typeof response.answer === 'string') {
  botResponse = response.answer;
} else if ('output' in response && typeof response.output === 'string') {
  botResponse = response.output;
} else if ('result' in response && typeof response.result === 'string') {
  botResponse = response.result;
} else if (typeof response === 'string') {
  botResponse = response;
} else if (response && typeof response === 'object') {
  // Try to find any property that might contain the answer
  const responseObj = response as Record<string, unknown>;
  const possibleAnswerKeys = Object.keys(responseObj).filter(key => 
    typeof responseObj[key] === 'string' && (responseObj[key] as string).length > 0
  );
  
  if (possibleAnswerKeys.length > 0) {
    botResponse = responseObj[possibleAnswerKeys[0]] as string;
  }
}
        
        addMessage(botResponse, 'bot');
      } else {
        // If no document is loaded yet
        addMessage("Please upload a document first so I can answer questions about it.", 'bot');
      }
    } catch (error) {
      console.error("Error getting response:", error);
      addMessage("Sorry, I encountered an error while processing your question. Please try again.", 'bot');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isApiKeySet) {
    return (
      <div className="chat-container">
        <div className="chat-header">
          <h1>JV AND DH2 RAG Chatbot</h1>
        </div>
        <div className="api-key-form">
          <p>Please enter your OpenAI API key to continue:</p>
          <form onSubmit={handleApiKeySubmit}>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your OpenAI API key"
              className="api-key-input"
            />
            <button type="submit">Submit</button>
          </form>
          <p className="api-key-note">Note: Your API key is stored locally in your browser and never sent to our servers.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>RAG Chatbot</h1>
      </div>
      
      <div className="file-upload">
        <label htmlFor="document-upload" className="file-upload-label">
          Upload Document
          <input
            type="file"
            id="document-upload"
            accept=".txt,.pdf,.doc,.docx"
            onChange={handleFileUpload}
            disabled={isLoading}
          />
        </label>
      </div>
      
      <div className="messages-container">
        {messages.map(message => (
          <div 
            key={message.id} 
            className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
          >
            {message.text}
          </div>
        ))}
        {isLoading && (
          <div className="message bot-message loading">
            <div className="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="input-form" onSubmit={handleSendMessage}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something about the document..."
          disabled={isLoading || !vectorStore}
        />
        <button type="submit" disabled={isLoading || !vectorStore}>
          Send
        </button>
      </form>
    </div>
  );
};

export default App;