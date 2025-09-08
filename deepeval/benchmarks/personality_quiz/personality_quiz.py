import os
import random
import pandas as pd
from typing import List, Dict, Optional, Any
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.test_case import LLMTestCase
from deepeval.benchmarks.personality_quiz.task import PersonalityQuizTask
from deepeval.benchmarks.personality_quiz.template import PersonalityQuizTemplate
from deepeval.benchmarks.personality_quiz.schema import PersonalityResponseSchema
from deepeval.benchmarks.personality_quiz.result import PersonalityQuizResult


class PersonalityQuiz(DeepEvalBaseBenchmark):
    """Personality Quiz benchmark using IPIP-NEO-120 questionnaire.
    
    This benchmark evaluates personality traits using the Big Five model:
    - Neuroticism
    - Extraversion
    - Openness
    - Agreeableness
    - Conscientiousness
    """
    
    def __init__(
        self,
        tasks: List[PersonalityQuizTask] = None,
        n_problems_per_task: int = 4,
        temperature: float = 0.0,
        random_seed: Optional[int] = None
    ):
        """Initialize the Personality Quiz benchmark.
        
        Args:
            tasks: List of personality traits to evaluate (default: all traits)
            n_problems_per_task: Number of questions per trait (default: 4)
            temperature: Temperature for model generation (default: 0.0)
            random_seed: Random seed for reproducible sampling (default: None)
        """
        super().__init__()
        
        self.tasks = tasks or list(PersonalityQuizTask)
        self.n_problems_per_task = n_problems_per_task
        self.temperature = temperature
        self.random_seed = random_seed
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
        
        # Load and validate dataset
        self.dataset = self._load_dataset()
        self._validate_dataset()
        
        # Prepare test cases
        self.test_cases = self._prepare_test_cases()
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load the IPIP-NEO-120 dataset from CSV."""
        # Get the path to the CSV file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "ipip_neo_120.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        
        # Load the CSV
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['trait', 'sub_facet', 'item_id', 'statement', 'keyed', 'domain']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def _validate_dataset(self) -> None:
        """Validate the loaded dataset structure and content."""
        # Check total number of items
        if len(self.dataset) != 120:
            raise ValueError(f"Expected 120 items, got {len(self.dataset)}")
        
        # Check trait distribution
        trait_counts = self.dataset['trait'].value_counts()
        expected_traits = ['Neuroticism', 'Extraversion', 'Openness', 'Agreeableness', 'Conscientiousness']
        
        for trait in expected_traits:
            if trait not in trait_counts:
                raise ValueError(f"Missing trait: {trait}")
            if trait_counts[trait] != 24:
                raise ValueError(f"Expected 24 items for {trait}, got {trait_counts[trait]}")
        
        # Check keying values
        unique_keyed = set(self.dataset['keyed'].unique())
        if unique_keyed != {1, -1}:
            raise ValueError(f"Expected keyed values to be 1 and -1, got {unique_keyed}")
        
        # Check item IDs are unique
        if len(self.dataset['item_id'].unique()) != 120:
            raise ValueError("Item IDs are not unique")
    
    def _prepare_test_cases(self) -> List[LLMTestCase]:
        """Prepare test cases by sampling questions proportionally across traits."""
        test_cases = []
        
        # Calculate total questions needed
        total_questions = len(self.tasks) * self.n_problems_per_task
        
        # Sample questions proportionally across traits
        for task in self.tasks:
            trait_name = task.value.title()  # Convert to title case for matching
            
            # Get all items for this trait
            trait_items = self.dataset[self.dataset['trait'] == trait_name]
            
            if len(trait_items) == 0:
                raise ValueError(f"No items found for trait: {trait_name}")
            
            # Sample items for this trait
            n_items = min(self.n_problems_per_task, len(trait_items))
            sampled_items = trait_items.sample(n=n_items, random_state=self.random_seed)
            
            # Create test cases
            for _, item in sampled_items.iterrows():
                # Format the question
                question = PersonalityQuizTemplate.format_question(item['statement'])
                
                # Create test case
                test_case = LLMTestCase(
                    input=question,
                    expected_output=str(item['keyed']),  # Store keying for scoring
                    context=[
                        f"Trait: {item['trait']}",
                        f"Sub-facet: {item['sub_facet']}",
                        f"Item ID: {item['item_id']}",
                        f"Keyed: {item['keyed']}",
                        f"Domain: {item['domain']}"
                    ]
                )
                test_cases.append(test_case)
        
        return test_cases
    
    def load_benchmark_dataset(self) -> List[LLMTestCase]:
        """Load the benchmark dataset.
        
        Returns:
            List of LLMTestCase objects for the personality quiz
        """
        return self.test_cases
    
    def evaluate(self, model) -> PersonalityQuizResult:
        """Evaluate the model on the personality quiz.
        
        Args:
            model: The model to evaluate
            
        Returns:
            PersonalityQuizResult containing trait scores and responses
        """
        responses = []
        total_cost = 0.0
        
        print(f"🧠 Evaluating model on {len(self.test_cases)} personality questions...")
        
        for i, test_case in enumerate(self.test_cases):
            try:
                # Generate response with schema
                response, cost = model.generate(
                    test_case.input,
                    schema=PersonalityResponseSchema
                )
                
                total_cost += cost
                
                # Store response data
                response_data = {
                    'test_case': test_case,
                    'response': response,
                    'cost': cost,
                    'index': i
                }
                responses.append(response_data)
                
                print(f"  ✅ Question {i+1}/{len(self.test_cases)}: Rating {response.rating}")
                
            except Exception as e:
                print(f"  ❌ Question {i+1}/{len(self.test_cases)}: Error - {e}")
                # Create a fallback response for error cases
                fallback_response = PersonalityResponseSchema(rating=3, confidence=None)
                response_data = {
                    'test_case': test_case,
                    'response': fallback_response,
                    'cost': 0.0,
                    'index': i,
                    'error': str(e)
                }
                responses.append(response_data)
        
        # Calculate trait scores
        trait_scores = self._calculate_trait_scores(responses)
        
        print(f"📊 Trait scores: {trait_scores}")
        print(f"💰 Total cost: ${total_cost:.4f}")
        
        result = PersonalityQuizResult(
            trait_scores=trait_scores,
            responses=responses,
            overall_accuracy=1.0,  # Placeholder - not applicable for personality scoring
            cost=total_cost
        )
        
        # Store result for later access
        self._last_result = result
        
        return result
    
    def _calculate_trait_scores(self, responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trait scores from model responses.
        
        Args:
            responses: List of response dictionaries containing test cases and model responses
            
        Returns:
            Dictionary mapping trait names to their calculated scores
        """
        trait_scores = {}
        
        for task in self.tasks:
            trait_name = task.value
            trait_responses = []
            
            # Filter responses for this trait
            for response in responses:
                test_case = response['test_case']
                trait_context = test_case.context[0]  # "Trait: Neuroticism"
                if trait_context.startswith(f'Trait: {trait_name.title()}'):
                    trait_responses.append(response)
            
            # Calculate score for this trait
            total_score = 0.0
            for response in trait_responses:
                # Handle both schema objects and fallback responses
                if hasattr(response['response'], 'rating'):
                    rating = response['response'].rating
                else:
                    # Fallback for error cases
                    rating = 3
                
                keyed = int(response['test_case'].expected_output)
                
                # Apply reverse scoring if needed (keyed = -1)
                if keyed == -1:
                    rating = 6 - rating
                
                total_score += rating
            
            trait_scores[trait_name] = total_score
        
        return trait_scores
    
    def get_trait_scores(self) -> Dict[str, float]:
        """Get trait scores from the test cases.
        
        Note: This method requires the benchmark to be evaluated first.
        
        Returns:
            Dictionary mapping trait names to scores
        """
        if not hasattr(self, '_last_result') or self._last_result is None:
            raise ValueError("Benchmark must be evaluated first. Call evaluate() method.")
        
        return self._last_result.get_trait_scores()
