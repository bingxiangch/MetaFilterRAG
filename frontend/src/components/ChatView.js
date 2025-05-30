import React, { useState } from 'react';
import api from '../common/api';
import { useChat } from './ChatContext';
import { BASE_URL } from '../config';
export const ChatView = () => {
  const { messages, loading, error, addMessage, setLoading, setError } = useChat();
  const [newMessage, setNewMessage] = useState('');
  // const [selectedMode, setSelectedMode] = useState('local'); // Default mode is OpenAI

  const handleSendMessage = async () => {
    if (newMessage.trim() !== '') {
      addMessage({ text: newMessage, sender: 'user' });
      setNewMessage('');
      setLoading(true);

      try {
        const response = await api.post(`http://localhost:8000/query`, { prompt: newMessage });
        addMessage({ text: response.data.answer, query_filter: response.data.query_filter, sender: 'bot' });
      } catch (error) {
        handleApiError(error);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleApiError = (error) => {
    if (error.response && error.response.status === 401) {
      setError('Unauthorized: Please log in again.');
    } else {
      setError(`Error: ${error.message}`);
    }

    console.error('Error fetching response from the server:', error.message);
    setTimeout(() => {
      setError(null);
    }, 3000);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !loading) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  // const handleModeChange = async (mode) => {
  //   setSelectedMode(mode);
  //   try {
  //     await api.put(`${BASE_URL}edit-llm-mode/`, { mode });
  //   } catch (error) {
  //     console.error('Error setting LLM mode:', error);
  //   }
  // };
  return (
    <main className="bg-slate-50 p-6 sm:p-10 flex-auto">
      <div>
        {error && <div style={{ color: 'red' }}>{error}</div>}
        <div style={{ height: '600px', overflowY: 'auto', border: '1px solid #ccc', padding: '10px' }}>
  {messages.map((message, index) => (
    <div key={index} style={{ marginBottom: '10px', textAlign: message.sender === 'user' ? 'right' : 'left' }}>
      <strong>{message.sender === 'user' ? 'You' : 'Bot'}:</strong> 
      <div dangerouslySetInnerHTML={{ __html: message.text.replace(/(?:\r\n|\r)(?![\n*])/g, '<br />').replace(/<ol>/g, '<ol class="list-decimal" style="padding-left: 20px;">') }} />
      {message.sender === 'bot' && message.query_filter && message.query_filter !== '{}' && (
  <div>Extracted_filter: {message.query_filter}</div>
)}

    </div>
  ))}
</div>

        <div className="flex items-center mt-4">
        {/* <span className="ml-2 text-gray-500">LLM Mode:</span>
          <select
            value={selectedMode}
            onChange={(e) => handleModeChange(e.target.value)}
            className="border rounded-r py-2 px-4 focus:outline-none"
          >
            <option value="local">Mistral-7b</option>
            <option value="openai">GPT3.5</option>
          </select> */}
          <input
            type="text"
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            className="border rounded-l py-2 px-4 w-3/4 focus:outline-none"
          />
          <button
            onClick={handleSendMessage}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-r focus:outline-none"
            disabled={loading}
          >
            {loading ? 'Waiting...' : 'Send'}
          </button>
        </div>
      </div>

    </main>
  );
};