from typing import List, Optional, Dict
from tqdm import tqdm
import re
import math
from collections import Counter

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import (
    DeepEvalBaseBenchmark,
    DeepEvalBaseBenchmarkResult,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.telemetry import capture_benchmark_run

from .schema import (
    RareWordAnalysis, 
    RareWordUsageResult, 
    RareWordUsageBenchmarkResult,
    GenerationPrompt
)


class RareWordUsageBenchmark(DeepEvalBaseBenchmark):
    """
    Benchmark that evaluates how frequently models use rare words in their generation.
    
    Uses a comprehensive English word frequency dataset from HuggingFace (173k words)
    to determine what constitutes a "rare" word, then evaluates model responses
    across diverse generation prompts spanning essays, science, mathematics, literature, 
    philosophy, coding, and creative writing.
    """
    
    def __init__(
        self,
        rare_word_threshold: int = 60000,  # Words ranked below this are considered rare (out of 173k)
        n_prompts: Optional[int] = None,  # Limit number of prompts (default: all ~200)
        categories: Optional[List[str]] = None,  # Filter by categories
        verbose_mode: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rare_word_threshold = rare_word_threshold
        self.n_prompts = n_prompts
        self.categories = categories
        self.verbose_mode = verbose_mode
        
        # Word frequency data will be loaded dynamically
        self.word_freq_map: Optional[Dict[str, int]] = None
        self.max_frequency_rank = 173000  # HuggingFace dataset has ~173k words
        
        # Load generation prompts
        self.prompts = self._load_generation_prompts()
    
    def _load_word_frequency_data(self) -> Dict[str, int]:
        """Load word frequency data from HuggingFace dataset"""
        return self._load_huggingface_frequency_data()
    
    def _load_huggingface_frequency_data(self) -> Dict[str, int]:
        """Load English word frequency data from HuggingFace dataset"""
        try:
            from datasets import load_dataset
            
            if self.verbose_mode:
                print("Loading English word frequency data from HuggingFace...")
            
            # Load the dataset - using the sorted_by_frequency subset
            dataset = load_dataset("Maximax67/English-Valid-Words", "sorted_by_frequency", split="train")
            
            word_freq_map = {}
            
            # Convert to frequency rank mapping
            for i, item in enumerate(dataset):
                word_raw = item["Word"]
                # Skip None values or empty strings
                if word_raw is None or not word_raw.strip():
                    continue
                    
                word = word_raw.lower().strip()
                # Skip if empty after processing
                if not word:
                    continue
                    
                # Rank is the position in frequency-sorted list (1-based)
                rank = i + 1  
                word_freq_map[word] = rank
            
            if self.verbose_mode:
                print(f"Successfully loaded {len(word_freq_map)} words from HuggingFace dataset")
                print(f"Dataset contains words ranked from 1 to {len(word_freq_map)}")
            
            self.max_frequency_rank = len(word_freq_map)
            return word_freq_map
            
        except ImportError as e:
            raise RuntimeError(f"Failed to import datasets library. Please install with: pip install datasets\nError: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace dataset: {e}")
    
    def _load_generation_prompts(self) -> List[GenerationPrompt]:
        """Load diverse generation prompts for testing rare word usage"""
        # For now, we'll define prompts inline. In a production system,
        # these could be loaded from HuggingFace datasets or external sources
        
        prompts = [
            # Academic Essays (20 prompts)
            GenerationPrompt(
                prompt_id="essay_001",
                category="essay",
                difficulty="medium",
                prompt_text="Write an essay about how scientific discoveries change the way we think about the world.",
                expected_response_length="long",
                description="General science and philosophy essay"
            ),
            GenerationPrompt(
                prompt_id="essay_002", 
                category="essay",
                difficulty="medium",
                prompt_text="Discuss the importance of preserving old books and manuscripts for future generations.",
                expected_response_length="long",
                description="Essay about cultural preservation"
            ),
            GenerationPrompt(
                prompt_id="essay_003",
                category="essay", 
                difficulty="medium",
                prompt_text="Explain how the design of buildings affects the way people feel and behave.",
                expected_response_length="long",
                description="Architecture and psychology essay"
            ),
            GenerationPrompt(
                prompt_id="essay_004",
                category="essay",
                difficulty="medium", 
                prompt_text="Write about whether fictional characters can be considered 'real' in some sense.",
                expected_response_length="long",
                description="Philosophy and literature essay"
            ),
            GenerationPrompt(
                prompt_id="essay_005",
                category="essay",
                difficulty="medium",
                prompt_text="Discuss different ways historians study the past and why their methods matter.",
                expected_response_length="long", 
                description="History and methodology essay"
            ),
            
            # Scientific Writing (25 prompts)
            GenerationPrompt(
                prompt_id="science_001",
                category="science",
                difficulty="medium",
                prompt_text="Explain how enzymes work and why they're important for living things.",
                expected_response_length="medium",
                description="Basic biochemistry explanation"
            ),
            GenerationPrompt(
                prompt_id="science_002",
                category="science", 
                difficulty="medium",
                prompt_text="Describe how the brain changes when we learn new things.",
                expected_response_length="medium",
                description="Neuroscience and learning explanation"
            ),
            GenerationPrompt(
                prompt_id="science_003",
                category="science",
                difficulty="medium",
                prompt_text="Explain how proteins fold into their proper shapes and why this matters.",
                expected_response_length="medium", 
                description="Protein biology explanation"
            ),
            GenerationPrompt(
                prompt_id="science_004",
                category="science",
                difficulty="medium",
                prompt_text="Describe a scientific technique used to study molecules and what it tells us.",
                expected_response_length="medium",
                description="Scientific methods explanation"
            ),
            GenerationPrompt(
                prompt_id="science_005",
                category="science",
                difficulty="medium",
                prompt_text="Explain why our bodies have internal clocks and how they work.",
                expected_response_length="medium",
                description="Circadian biology explanation"
            ),
            
            # Creative Writing (30 prompts)
            GenerationPrompt(
                prompt_id="creative_001",
                category="creative",
                difficulty="medium", 
                prompt_text="Write a short story about someone who discovers that maps can change the real world.",
                expected_response_length="long",
                description="Fantasy/science fiction story"
            ),
            GenerationPrompt(
                prompt_id="creative_002",
                category="creative",
                difficulty="medium",
                prompt_text="Write a poem about longing for something that's lost forever.",
                expected_response_length="medium", 
                description="Emotional poetry about loss"
            ),
            GenerationPrompt(
                prompt_id="creative_003",
                category="creative",
                difficulty="medium",
                prompt_text="Tell a story about someone who collects old, forgotten words.",
                expected_response_length="long",
                description="Story about language and preservation"
            ),
            GenerationPrompt(
                prompt_id="creative_004", 
                category="creative",
                difficulty="medium",
                prompt_text="Write a conversation between two people discussing what it means to experience color.",
                expected_response_length="medium",
                description="Dialogue about perception and consciousness"
            ),
            GenerationPrompt(
                prompt_id="creative_005",
                category="creative",
                difficulty="medium",
                prompt_text="Create a story where abstract ideas like numbers or shapes become physical things.",
                expected_response_length="long", 
                description="Fantasy story with mathematical elements"
            ),
            
            # Technical/Coding (20 prompts) 
            GenerationPrompt(
                prompt_id="coding_001",
                category="coding",
                difficulty="medium",
                prompt_text="Explain how to create a data structure that remembers previous versions of itself.",
                expected_response_length="medium",
                description="Computer science and data structures"
            ),
            GenerationPrompt(
                prompt_id="coding_002",
                category="coding",
                difficulty="medium",
                prompt_text="Describe how multiple computers can agree on something even when some might be broken.",
                expected_response_length="medium",
                description="Distributed systems concepts"
            ),
            GenerationPrompt(
                prompt_id="coding_003", 
                category="coding",
                difficulty="medium",
                prompt_text="Explain how programming languages automatically clean up unused memory.",
                expected_response_length="medium",
                description="Memory management in programming"
            ),
            GenerationPrompt(
                prompt_id="coding_004",
                category="coding", 
                difficulty="medium",
                prompt_text="Describe how a computer program gets converted from code into something the machine can run.",
                expected_response_length="medium",
                description="Compilation process explanation"
            ),
            GenerationPrompt(
                prompt_id="coding_005",
                category="coding",
                difficulty="medium",
                prompt_text="Explain how modern encryption works and why it's secure.",
                expected_response_length="medium",
                description="Cryptography and security basics"
            ),
            
            # Mathematics (25 prompts)
            GenerationPrompt(
                prompt_id="math_001", 
                category="mathematics",
                difficulty="medium",
                prompt_text="Explain what happens when you try to untangle a complicated knot and why some knots are harder than others.",
                expected_response_length="medium",
                description="Topology and knot theory in accessible terms"
            ),
            GenerationPrompt(
                prompt_id="math_002",
                category="mathematics",
                difficulty="medium", 
                prompt_text="Describe why some mathematical equations can't be solved using simple formulas.",
                expected_response_length="medium",
                description="Abstract algebra concepts"
            ),
            GenerationPrompt(
                prompt_id="math_003",
                category="mathematics",
                difficulty="medium",
                prompt_text="Explain a mathematical theorem that connects geometry with other areas of math.",
                expected_response_length="medium",
                description="Mathematical connections and applications"
            ),
            GenerationPrompt(
                prompt_id="math_004",
                category="mathematics", 
                difficulty="medium",
                prompt_text="Discuss how mathematicians organize and classify different types of mathematical objects.",
                expected_response_length="medium",
                description="Mathematical classification systems"
            ),
            GenerationPrompt(
                prompt_id="math_005",
                category="mathematics",
                difficulty="medium",
                prompt_text="Explain how mathematicians study systems that change over time in predictable ways.",
                expected_response_length="medium", 
                description="Dynamical systems and patterns"
            ),
            
            # More Essays (15 prompts)
            GenerationPrompt(
                prompt_id="essay_006",
                category="essay",
                difficulty="medium",
                prompt_text="Write about the role of art in society and why it matters to people.",
                expected_response_length="long",
                description="Art and society essay"
            ),
            GenerationPrompt(
                prompt_id="essay_007",
                category="essay",
                difficulty="medium",
                prompt_text="Discuss how technology has changed the way we communicate with each other.",
                expected_response_length="long",
                description="Technology and communication essay"
            ),
            GenerationPrompt(
                prompt_id="essay_008",
                category="essay",
                difficulty="medium",
                prompt_text="Explain why education systems around the world are so different.",
                expected_response_length="long",
                description="Education and culture essay"
            ),
            GenerationPrompt(
                prompt_id="essay_009",
                category="essay",
                difficulty="medium",
                prompt_text="Write about how cities develop and change over time.",
                expected_response_length="long",
                description="Urban development essay"
            ),
            GenerationPrompt(
                prompt_id="essay_010",
                category="essay",
                difficulty="medium",
                prompt_text="Discuss the relationship between individual freedom and social responsibility.",
                expected_response_length="long",
                description="Ethics and society essay"
            ),
            
            # More Science (15 prompts)
            GenerationPrompt(
                prompt_id="science_006",
                category="science",
                difficulty="medium",
                prompt_text="Explain how scientists study things that are too small to see.",
                expected_response_length="medium",
                description="Microscopy and molecular science"
            ),
            GenerationPrompt(
                prompt_id="science_007",
                category="science",
                difficulty="medium",
                prompt_text="Describe how living things adapt to their environment over time.",
                expected_response_length="medium",
                description="Evolution and adaptation"
            ),
            GenerationPrompt(
                prompt_id="science_008",
                category="science",
                difficulty="medium",
                prompt_text="Explain how the weather works and why it's hard to predict.",
                expected_response_length="medium",
                description="Meteorology and climate"
            ),
            GenerationPrompt(
                prompt_id="science_009",
                category="science",
                difficulty="medium",
                prompt_text="Describe what happens inside stars and how they affect space around them.",
                expected_response_length="medium",
                description="Astronomy and stellar physics"
            ),
            GenerationPrompt(
                prompt_id="science_010",
                category="science",
                difficulty="medium",
                prompt_text="Explain how doctors use technology to see inside the human body.",
                expected_response_length="medium",
                description="Medical imaging and technology"
            ),
            
            # More Creative Writing (15 prompts)
            GenerationPrompt(
                prompt_id="creative_006",
                category="creative",
                difficulty="medium",
                prompt_text="Write a story about someone who can hear other people's thoughts.",
                expected_response_length="long",
                description="Science fiction story about telepathy"
            ),
            GenerationPrompt(
                prompt_id="creative_007",
                category="creative",
                difficulty="medium",
                prompt_text="Create a poem about the feeling of being in a place you've never been before.",
                expected_response_length="medium",
                description="Poetry about new experiences"
            ),
            GenerationPrompt(
                prompt_id="creative_008",
                category="creative",
                difficulty="medium",
                prompt_text="Tell a story about a world where colors have different meanings than they do now.",
                expected_response_length="long",
                description="Fantasy story about perception"
            ),
            GenerationPrompt(
                prompt_id="creative_009",
                category="creative",
                difficulty="medium",
                prompt_text="Write about a character who discovers they have an unusual talent.",
                expected_response_length="long",
                description="Character development story"
            ),
            GenerationPrompt(
                prompt_id="creative_010",
                category="creative",
                difficulty="medium",
                prompt_text="Create a dialogue between a child and an elderly person about time.",
                expected_response_length="medium",
                description="Intergenerational dialogue"
            ),
            
            # Philosophy and Ideas (15 prompts)
            GenerationPrompt(
                prompt_id="philosophy_001",
                category="philosophy",
                difficulty="medium",
                prompt_text="Discuss what makes something beautiful and whether beauty is the same for everyone.",
                expected_response_length="medium",
                description="Aesthetics and philosophy of beauty"
            ),
            GenerationPrompt(
                prompt_id="philosophy_002",
                category="philosophy",
                difficulty="medium",
                prompt_text="Explain what it means to know something and how we can be sure our knowledge is correct.",
                expected_response_length="medium",
                description="Epistemology and knowledge"
            ),
            GenerationPrompt(
                prompt_id="philosophy_003",
                category="philosophy",
                difficulty="medium",
                prompt_text="Write about whether we have free will or if everything we do is determined by prior causes.",
                expected_response_length="medium",
                description="Free will and determinism"
            ),
            GenerationPrompt(
                prompt_id="philosophy_004",
                category="philosophy",
                difficulty="medium",
                prompt_text="Discuss what makes an action right or wrong and who decides.",
                expected_response_length="medium",
                description="Ethics and morality"
            ),
            GenerationPrompt(
                prompt_id="philosophy_005",
                category="philosophy",
                difficulty="medium",
                prompt_text="Explain the relationship between the mind and the physical brain.",
                expected_response_length="medium",
                description="Philosophy of mind"
            ),
            
            # General Knowledge and Current Topics (15 prompts)
            GenerationPrompt(
                prompt_id="knowledge_001",
                category="knowledge",
                difficulty="medium",
                prompt_text="Explain how global trade affects local communities around the world.",
                expected_response_length="medium",
                description="Economics and globalization"
            ),
            GenerationPrompt(
                prompt_id="knowledge_002",
                category="knowledge",
                difficulty="medium",
                prompt_text="Describe how different cultures approach the concept of family.",
                expected_response_length="medium",
                description="Anthropology and cultural studies"
            ),
            GenerationPrompt(
                prompt_id="knowledge_003",
                category="knowledge",
                difficulty="medium",
                prompt_text="Discuss how social media has changed human relationships.",
                expected_response_length="medium",
                description="Sociology and technology"
            ),
            GenerationPrompt(
                prompt_id="knowledge_004",
                category="knowledge",
                difficulty="medium",
                prompt_text="Explain how governments make decisions that affect millions of people.",
                expected_response_length="medium",
                description="Political science and governance"
            ),
            GenerationPrompt(
                prompt_id="knowledge_005",
                category="knowledge",
                difficulty="medium",
                prompt_text="Describe how humans impact the natural environment and what can be done about it.",
                expected_response_length="medium",
                description="Environmental science and policy"
            ),
            
            # More Technical/Coding (10 prompts)
            GenerationPrompt(
                prompt_id="coding_006",
                category="coding",
                difficulty="medium",
                prompt_text="Explain how websites know who you are when you visit them.",
                expected_response_length="medium",
                description="Web authentication and sessions"
            ),
            GenerationPrompt(
                prompt_id="coding_007",
                category="coding",
                difficulty="medium",
                prompt_text="Describe how computers can learn to recognize patterns in data.",
                expected_response_length="medium",
                description="Machine learning basics"
            ),
            GenerationPrompt(
                prompt_id="coding_008",
                category="coding",
                difficulty="medium",
                prompt_text="Explain why some computer programs run faster than others.",
                expected_response_length="medium",
                description="Performance and optimization"
            ),
            GenerationPrompt(
                prompt_id="coding_009",
                category="coding",
                difficulty="medium",
                prompt_text="Describe how programmers work together on large software projects.",
                expected_response_length="medium",
                description="Software engineering and collaboration"
            ),
            GenerationPrompt(
                prompt_id="coding_010",
                category="coding",
                difficulty="medium",
                prompt_text="Explain how computers store and organize large amounts of information.",
                expected_response_length="medium",
                description="Databases and data storage"
            ),
            
            # More Mathematics (10 prompts)
            GenerationPrompt(
                prompt_id="math_006",
                category="mathematics",
                difficulty="medium",
                prompt_text="Explain why some infinities are bigger than others.",
                expected_response_length="medium",
                description="Set theory and infinity"
            ),
            GenerationPrompt(
                prompt_id="math_007",
                category="mathematics",
                difficulty="medium",
                prompt_text="Describe how mathematicians use probability to understand uncertainty.",
                expected_response_length="medium",
                description="Probability and statistics"
            ),
            GenerationPrompt(
                prompt_id="math_008",
                category="mathematics",
                difficulty="medium",
                prompt_text="Explain how geometric shapes relate to algebraic equations.",
                expected_response_length="medium",
                description="Analytic geometry"
            ),
            GenerationPrompt(
                prompt_id="math_009",
                category="mathematics",
                difficulty="medium",
                prompt_text="Describe how mathematicians study change and motion.",
                expected_response_length="medium",
                description="Calculus concepts"
            ),
            GenerationPrompt(
                prompt_id="math_010",
                category="mathematics",
                difficulty="medium",
                prompt_text="Explain how patterns in numbers can reveal hidden mathematical truths.",
                expected_response_length="medium",
                description="Number theory and patterns"
            ),
            
            # Literature Analysis (20 prompts)
            GenerationPrompt(
                prompt_id="literature_001",
                category="literature",
                difficulty="medium",
                prompt_text="Discuss how authors use literary techniques to create meaning beyond the literal story.",
                expected_response_length="medium",
                description="Literary analysis and interpretation"
            ),
            GenerationPrompt(
                prompt_id="literature_002", 
                category="literature",
                difficulty="medium",
                prompt_text="Explain how different authors tell similar stories in their own unique ways.",
                expected_response_length="medium",
                description="Literary comparison and style"
            ),
            GenerationPrompt(
                prompt_id="literature_003",
                category="literature",
                difficulty="medium", 
                prompt_text="Discuss what happens in your mind when you read a book and how it affects your understanding.",
                expected_response_length="medium",
                description="Reading experience and interpretation"
            ),
            GenerationPrompt(
                prompt_id="literature_004",
                category="literature",
                difficulty="medium",
                prompt_text="Describe how novels can contain many different voices and perspectives at once.",
                expected_response_length="medium",
                description="Narrative techniques and multiple perspectives"
            ),
            GenerationPrompt(
                prompt_id="literature_005",
                category="literature", 
                difficulty="medium",
                prompt_text="Explain how writers can make familiar things seem strange and new.",
                expected_response_length="medium",
                description="Literary techniques and defamiliarization"
            ),
            
            # History (10 prompts)
            GenerationPrompt(
                prompt_id="history_001",
                category="history",
                difficulty="medium",
                prompt_text="Discuss how historians debate when one historical period ends and another begins.",
                expected_response_length="medium",
                description="Historical periodization and interpretation"
            ),
            GenerationPrompt(
                prompt_id="history_002", 
                category="history",
                difficulty="medium",
                prompt_text="Explain how historians study social connections and relationships in past societies.",
                expected_response_length="medium",
                description="Social history and networks"
            ),
            GenerationPrompt(
                prompt_id="history_003",
                category="history",
                difficulty="medium",
                prompt_text="Describe how historians study long-term changes that happen over centuries.",
                expected_response_length="medium", 
                description="Long-term historical change"
            ),
            GenerationPrompt(
                prompt_id="history_004",
                category="history",
                difficulty="medium",
                prompt_text="Explain how old handwritten documents tell us about the past.",
                expected_response_length="medium",
                description="Historical sources and manuscripts"
            ),
            GenerationPrompt(
                prompt_id="history_005",
                category="history", 
                difficulty="medium",
                prompt_text="Discuss how coins and money can reveal information about ancient economies.",
                expected_response_length="medium",
                description="Economic history through material evidence"
            )
        ]
        
        # Filter by categories if specified
        if self.categories:
            prompts = [p for p in prompts if p.category in self.categories]
        
        # Limit number of prompts if specified
        if self.n_prompts:
            prompts = prompts[:self.n_prompts]
        
        return prompts

    def load_benchmark_dataset(self) -> List[Golden]:
        """Convert generation prompts to Golden objects"""
        goldens = []
        for prompt in self.prompts:
            golden = Golden(
                input=prompt.prompt_text,
                expected_output="",  # No expected output for generation tasks
                additional_metadata={
                    "prompt_id": prompt.prompt_id,
                    "category": prompt.category,
                    "difficulty": prompt.difficulty,
                    "expected_response_length": prompt.expected_response_length,
                    "description": prompt.description,
                }
            )
            goldens.append(golden)
        return goldens

    def evaluate(
        self, 
        model: DeepEvalBaseLLM,
        **kwargs
    ) -> DeepEvalBaseBenchmarkResult:
        """Evaluate model for rare word usage across diverse generation prompts"""
        
        # Load word frequency data if not already loaded
        if self.word_freq_map is None:
            self.word_freq_map = self._load_word_frequency_data()
        
        with capture_benchmark_run("RareWordUsage", len(self.prompts)):
            goldens = self.load_benchmark_dataset()
            
            total_rare_word_score = 0.0
            total_words_generated = 0
            total_rare_words = 0
            category_scores = {}
            category_counts = {}
            detailed_results = []
            
            for golden in tqdm(
                goldens, 
                desc=f"Evaluating rare word usage with {model.get_model_name()}"
            ):
                # Generate response - consistent with other benchmarks
                response, _ = model.generate(golden.input)
                
                # Analyze rare word usage
                result = self._analyze_rare_word_usage(response)
                
                # Track overall statistics
                total_rare_word_score += result.average_log_frequency
                total_words_generated += result.total_words
                total_rare_words += result.rare_words_count
                
                # Track category statistics
                category = golden.additional_metadata["category"]
                if category not in category_scores:
                    category_scores[category] = 0.0
                    category_counts[category] = 0
                
                category_scores[category] += result.average_log_frequency
                category_counts[category] += 1
                
                # Store detailed result
                detailed_result = {
                    "prompt_id": golden.additional_metadata["prompt_id"],
                    "category": category,
                    "difficulty": golden.additional_metadata["difficulty"],
                    "response": response,
                    "rare_word_analysis": result.dict(),
                }
                detailed_results.append(detailed_result)
                
                if self.verbose_mode:
                    print(f"\nPrompt {golden.additional_metadata['prompt_id']} ({category}):")
                    print(f"Rare words: {result.rare_words_count}/{result.total_words} ({result.rare_words_percentage:.1f}%)")
                    print(f"Avg log frequency: {result.average_log_frequency:.3f}")
                    if result.rare_words_found:
                        rare_words = [rw.word for rw in result.rare_words_found[:5]]
                        print(f"Sample rare words: {', '.join(rare_words)}")
                    print("-" * 50)
            
            # Calculate overall scores
            overall_rare_word_score = total_rare_word_score / len(goldens)
            average_rare_word_percentage = (total_rare_words / total_words_generated) * 100
            
            # Calculate category averages
            category_breakdown = {}
            for category in category_scores:
                category_breakdown[category] = category_scores[category] / category_counts[category]
            
            if self.verbose_mode:
                print(f"\n{'='*60}")
                print(f"RARE WORD USAGE RESULTS")
                print(f"{'='*60}")
                print(f"Model: {model.get_model_name()}")
                print(f"Total prompts: {len(goldens)}")
                print(f"Total words generated: {total_words_generated}")
                print(f"Total rare words: {total_rare_words}")
                print(f"Average rare word percentage: {average_rare_word_percentage:.2f}%")
                print(f"Overall rare word score: {overall_rare_word_score:.3f}")
                print(f"\nCategory breakdown:")
                for category, score in category_breakdown.items():
                    print(f"  {category}: {score:.3f}")
            
            return DeepEvalBaseBenchmarkResult(overall_accuracy=overall_rare_word_score)

    def _analyze_rare_word_usage(self, text: str) -> RareWordUsageResult:
        """Analyze the rare word usage in a given text"""
        # Tokenize text
        tokens = self._tokenize_text(text)
        
        # Filter to words only (remove punctuation, numbers)
        words = self._extract_words(tokens)
        
        if not words:
            return RareWordUsageResult(
                total_words=0,
                rare_words_count=0,
                rare_words_percentage=0.0,
                average_log_frequency=0.0,
                rare_words_found=[],
                unique_rare_words=0
            )
        
        # Analyze each word
        rare_words_found = []
        log_frequencies = []
        
        for word in words:
            word_lower = word.lower()
            
            if word_lower in self.word_freq_map:
                rank = self.word_freq_map[word_lower]
                is_rare = rank > self.rare_word_threshold
                log_freq = math.log(rank)
            else:
                # Unknown word - treat as very rare
                rank = None
                is_rare = True
                log_freq = math.log(self.max_frequency_rank + 1000)  # Penalty for unknown words
            
            if is_rare:
                analysis = RareWordAnalysis(
                    word=word,
                    frequency_rank=rank,
                    log_frequency=log_freq,
                    is_rare=True,
                    is_unknown=(rank is None)
                )
                rare_words_found.append(analysis)
                log_frequencies.append(log_freq)
        
        # Calculate statistics
        total_words = len(words)
        rare_words_count = len(rare_words_found)
        rare_words_percentage = (rare_words_count / total_words) * 100
        average_log_frequency = sum(log_frequencies) / len(log_frequencies) if log_frequencies else 0.0
        unique_rare_words = len(set(rw.word.lower() for rw in rare_words_found))
        
        return RareWordUsageResult(
            total_words=total_words,
            rare_words_count=rare_words_count,
            rare_words_percentage=rare_words_percentage,
            average_log_frequency=average_log_frequency,
            rare_words_found=rare_words_found,
            unique_rare_words=unique_rare_words
        )
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words using regex"""
        # Use regex to find word tokens (alphabetic characters only)
        tokens = re.findall(r'\b[a-zA-Z]+\b', text)
        return tokens
    
    def _extract_words(self, tokens: List[str]) -> List[str]:
        """Extract only words from tokens (remove single letters and short words)"""
        words = []
        for token in tokens:
            # Keep only words with 2+ characters
            if len(token) > 1:
                words.append(token)
        return words
