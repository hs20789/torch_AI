// app.js
const express = require('express');
const multer = require('multer');
const fs = require('fs').promises;
const crypto = require('crypto');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(express.static('public'));

const OLLAMA_URL = 'http://localhost:11434/api/generate';

// 진행 중인 분석 작업 저장
const analysisJobs = new Map();

async function analyzeNovel(text) {
  try {
    console.log('Sending request to Ollama...');

    const requestBody = {
      model: 'gemma3:1b',
      prompt: `
You are a story reviewer.

Analyze the following short novel.

Please provide:
1. A short summary
2. Main characters
3. Plot analysis
4. Character development
5. Emotional content
6. Suggestions for improvement

Novel:
${text}
`,
      stream: false,
    };

    // Node.js 18+에서는 fetch가 기본 제공됨
    const response = await fetch(OLLAMA_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`Ollama HTTP error: ${response.status}`);
    }

    const data = await response.json();

    console.log('Analysis completed.');

    return data.response;
  } catch (error) {
    console.error('Error in analyzeNovel:', error);
    throw error;
  }
}

app.post('/analyze', upload.single('novel'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'No file uploaded',
      });
    }

    // 충돌 가능성이 적은 고유 ID 생성
    const jobId = crypto.randomUUID();

    analysisJobs.set(jobId, {
      status: 'processing',
    });

    // 업로드된 txt 파일 읽기
    const fileContent = await fs.readFile(req.file.path, 'utf8');

    // 읽은 후 임시 업로드 파일 삭제
    await fs.unlink(req.file.path);

    // 클라이언트에 jobId 먼저 반환
    res.json({
      jobId,
    });

    // Ollama 분석 수행
    analyzeNovel(fileContent)
      .then((result) => {
        analysisJobs.set(jobId, {
          status: 'completed',
          result,
        });
      })
      .catch((error) => {
        analysisJobs.set(jobId, {
          status: 'error',
          error: error.message,
        });
      });
  } catch (error) {
    console.error('Error initiating analysis:', error);

    res.status(500).json({
      error: error.message,
    });
  }
});

// 분석 상태 조회
app.get('/status/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const job = analysisJobs.get(jobId);

  if (!job) {
    return res.status(404).json({
      error: 'Job not found',
    });
  }

  res.json(job);
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});