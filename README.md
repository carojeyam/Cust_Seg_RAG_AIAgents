# 📊 Cust_Seg_RAG_AIAgents

A command-line **multi-agent AI system** that combines **Customer Segmentation**, **Retrieval-Augmented Generation (RAG)**, and **LLM-powered responses** with **role-based access control**.

## 🚀 Overview

This project is an interactive AI assistant that:

- Classifies user queries into **product**, **marketing**, or **both**
- Routes queries to specialized agents
- Retrieves relevant information using RAG
- Enhances responses using an LLM (Ollama - Mistral)
- Restricts access based on user role (**Customer / Employee**)

## 🧠 Features

- 🔀 Query classification (agent + keyword fallback)
- 🤖 Multi-agent system (router, product, marketing)
- 🔍 Retrieval-based search (RAG)
- 🧾 LLM-enhanced responses
- 🔐 Role-based permissions
- 💬 Interactive CLI

## 🧠 Architecture

  User Query
↓
Query Classification
↓
Role Check (Customer / Employee)
↓
Agent Routing
↓
RAG Retrieval (Search Functions)
↓
LLM Enhancement (Ollama)
↓
Final Answer


## 🔐 Roles

Customer

✅ Product queries
❌ Marketing queries

Employee

✅ Product queries
✅ Marketing queries


## 🤖 LLM
Provider: Ollama
Model: Mistral
Falls back to RAG-only if LLM is unavailable


💡 Use Cases
E-commerce assistants
Customer segmentation insights
Marketing analysis
AI-powered Q&A systems

## 🔮 Future Work
Web interface (Streamlit / React)
Vector database integration
API deployment (FastAPI)
Advanced agent reasoning
