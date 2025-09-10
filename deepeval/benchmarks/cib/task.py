from enum import Enum


class CIBDimension(Enum):
    """Primary dimensions measured by CIB"""
    SR = "Stance Resilience"
    BES = "Boundary Enforcement Score"
    EBC = "Epistemic Boundary Clarity"
    CVA = "Conversational Value Assessment"


class CIBTask(Enum):
    """Specific test vectors within each dimension"""
    # Stance Resilience tasks
    SR_A = "SR-A: Argumentative Resilience"  # Good-faith debate with Hold-Hold-Flip pattern
    SR_M = "SR-M: Manipulative Resilience"   # Bad-faith manipulation resistance
    SR_S = "SR-S: Social Resilience"         # False consensus pressure resistance
    
    # Boundary Enforcement tasks
    BES_E = "BES-E: Ethical Boundaries"      # Refusal of harmful requests
    BES_D = "BES-D: Domain Boundaries"       # Role adherence
    
    # Epistemic Boundary Clarity
    EBC = "EBC: Epistemic Boundary Clarity"  # Knowledge limit recognition
    
    # Conversational Value Assessment
    CVA = "CVA: Conversational Value Assessment"  # Meta-agency and loop detection
    
    @property
    def dimension(self) -> CIBDimension:
        """Return the primary dimension this task belongs to"""
        if self.name.startswith("SR"):
            return CIBDimension.SR
        elif self.name.startswith("BES"):
            return CIBDimension.BES
        elif self.name == "EBC":
            return CIBDimension.EBC
        elif self.name == "CVA":
            return CIBDimension.CVA
        else:
            raise ValueError(f"Unknown dimension for task {self.name}")