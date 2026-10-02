const state = {
  model: null,
};

const form = document.querySelector("#prediction-form");
const submitButton = document.querySelector("#submit-button");
const salaryValue = document.querySelector("#salary-value");
const skillResults = document.querySelector("#skill-results");
const skillCount = document.querySelector("#skill-count");
const errorMessage = document.querySelector("#error-message");
const modelStatus = document.querySelector("#model-status");

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function characterNgrams(text, minimum, maximum) {
  const characters = Array.from(text);
  const counts = new Map();

  for (let size = minimum; size <= maximum; size += 1) {
    for (let index = 0; index <= characters.length - size; index += 1) {
      const token = characters.slice(index, index + size).join("");
      counts.set(token, (counts.get(token) || 0) + 1);
    }
  }
  return counts;
}

function transformVectorizer(text, vectorizer) {
  const vector = new Float64Array(vectorizer.idf.length);
  const counts = characterNgrams(
    text,
    vectorizer.ngram_min,
    vectorizer.ngram_max,
  );
  let squaredNorm = 0;

  counts.forEach((count, token) => {
    const index = vectorizer.vocabulary[token];
    if (index === undefined) {
      return;
    }
    const value = (1 + Math.log(count)) * vectorizer.idf[index];
    vector[index] = value;
    squaredNorm += value * value;
  });

  if (squaredNorm > 0) {
    const norm = Math.sqrt(squaredNorm);
    for (let index = 0; index < vector.length; index += 1) {
      vector[index] /= norm;
    }
  }
  return vector;
}

function dotProduct(vector, coefficients) {
  let score = 0;
  for (let index = 0; index < vector.length; index += 1) {
    if (vector[index] !== 0) {
      score += vector[index] * coefficients[index];
    }
  }
  return score;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, value))));
}

function buildInputText() {
  return normalizeText(
    [
      document.querySelector("#job-title").value,
      document.querySelector("#city").value,
      document.querySelector("#experience").value,
      document.querySelector("#education").value,
      document.querySelector("#job-description").value,
    ].join(" "),
  );
}

function predict(inputText) {
  const skillModel = state.model.skill_model;
  const salaryModel = state.model.salary_model;
  const skillVector = transformVectorizer(inputText, skillModel.vectorizer);
  const salaryVector = transformVectorizer(inputText, salaryModel.vectorizer);

  const skills = skillModel.labels
    .map((label, index) => ({
      label,
      probability: sigmoid(
        skillModel.intercepts[index] +
          dotProduct(skillVector, skillModel.coefficients[index]),
      ),
    }))
    .sort((left, right) => right.probability - left.probability);

  const rawSalary =
    salaryModel.intercept + dotProduct(salaryVector, salaryModel.coefficients);
  const salary = Math.max(
    salaryModel.minimum_k,
    Math.min(salaryModel.maximum_k, rawSalary),
  );

  return { skills, salary };
}

function renderSkills(skills) {
  const visibleSkills = skills
    .filter((item) => item.probability >= 0.24)
    .slice(0, 8);
  const fallbackSkills = skills.slice(0, 5);
  const resultSkills = visibleSkills.length ? visibleSkills : fallbackSkills;

  skillResults.innerHTML = resultSkills
    .map((item) => {
      const percentage = Math.round(item.probability * 100);
      return `
        <div class="skill-item">
          <strong>${item.label}</strong>
          <div class="skill-track">
            <div class="skill-fill" style="width: ${percentage}%"></div>
          </div>
          <span class="skill-value">${percentage}%</span>
        </div>
      `;
    })
    .join("");
  skillCount.textContent = `${resultSkills.length} 项技能`;
}

function runPrediction(event) {
  event.preventDefault();
  if (!state.model) {
    return;
  }

  errorMessage.textContent = "";
  submitButton.disabled = true;

  try {
    const result = predict(buildInputText());
    salaryValue.textContent = result.salary.toFixed(1);
    renderSkills(result.skills);
  } catch (error) {
    errorMessage.textContent = `预测失败：${error.message}`;
  } finally {
    submitButton.disabled = false;
  }
}

async function initialize() {
  try {
    const response = await fetch("model.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`模型加载失败 (${response.status})`);
    }
    state.model = await response.json();
    document.querySelector("#training-rows").textContent =
      state.model.dataset_rows;
    document.querySelector("#label-count").textContent =
      state.model.skill_model.labels.length;
    modelStatus.textContent = "可用";
    submitButton.disabled = false;
    runPrediction(new Event("submit"));
  } catch (error) {
    modelStatus.textContent = "加载失败";
    errorMessage.textContent = error.message;
  }
}

form.addEventListener("submit", runPrediction);
form.addEventListener("reset", () => {
  window.setTimeout(() => runPrediction(new Event("submit")), 0);
});

initialize();
