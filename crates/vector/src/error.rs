//! Errors produced when adding vectors to, or querying, a vector index.

use core::fmt;

/// The largest dimensionality a vector index will accept.
///
/// Well past any embedding model in use (OpenAI's largest is 3072, Cohere's 1024),
/// but small enough that `dimension * size_of::<f32>()` cannot overflow a bounds check
/// and that a malformed module definition cannot ask the host to allocate absurd rows.
pub const MAX_DIMENSION: usize = 16_384;

/// Something was wrong with a vector handed to a vector index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorError {
    /// The vector's length did not match the index's declared dimensionality.
    DimensionMismatch {
        /// The dimensionality the index was created with.
        expected: usize,
        /// The length of the offending vector.
        actual: usize,
    },

    /// The declared dimensionality was zero, or exceeded [`MAX_DIMENSION`].
    InvalidDimension {
        /// The rejected dimensionality.
        dimension: usize,
    },

    /// The vector contained a `NaN` or an infinity.
    ///
    /// Non-finite components make distances incomparable (`NaN` compares false against
    /// everything), which would silently corrupt a search. They are rejected at the door
    /// rather than allowed to poison an index.
    NonFinite {
        /// The index of the first offending component.
        position: usize,
    },
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "expected a vector of dimension {expected}, but got one of dimension {actual}"
                )
            }
            Self::InvalidDimension { dimension } => {
                write!(
                    f,
                    "invalid vector dimension {dimension}: must be between 1 and {MAX_DIMENSION} inclusive"
                )
            }
            Self::NonFinite { position } => {
                write!(
                    f,
                    "vector component at position {position} is not finite (NaN or infinity)"
                )
            }
        }
    }
}

impl std::error::Error for VectorError {}

/// Checks that `dimension` is a dimensionality a vector index can be created with.
pub fn validate_dimension(dimension: usize) -> Result<(), VectorError> {
    if dimension == 0 || dimension > MAX_DIMENSION {
        Err(VectorError::InvalidDimension { dimension })
    } else {
        Ok(())
    }
}

/// Checks that `vector` has length `dimension` and contains only finite components.
pub fn validate_vector(vector: &[f32], dimension: usize) -> Result<(), VectorError> {
    if vector.len() != dimension {
        return Err(VectorError::DimensionMismatch {
            expected: dimension,
            actual: vector.len(),
        });
    }
    match vector.iter().position(|c| !c.is_finite()) {
        Some(position) => Err(VectorError::NonFinite { position }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_bounds() {
        assert_eq!(
            validate_dimension(0),
            Err(VectorError::InvalidDimension { dimension: 0 })
        );
        assert_eq!(validate_dimension(1), Ok(()));
        assert_eq!(validate_dimension(MAX_DIMENSION), Ok(()));
        assert_eq!(
            validate_dimension(MAX_DIMENSION + 1),
            Err(VectorError::InvalidDimension {
                dimension: MAX_DIMENSION + 1
            })
        );
    }

    #[test]
    fn rejects_wrong_length() {
        assert_eq!(
            validate_vector(&[1.0, 2.0], 3),
            Err(VectorError::DimensionMismatch { expected: 3, actual: 2 })
        );
    }

    #[test]
    fn rejects_non_finite() {
        assert_eq!(
            validate_vector(&[1.0, f32::NAN, 3.0], 3),
            Err(VectorError::NonFinite { position: 1 })
        );
        assert_eq!(
            validate_vector(&[f32::INFINITY, 2.0], 2),
            Err(VectorError::NonFinite { position: 0 })
        );
        assert_eq!(
            validate_vector(&[1.0, f32::NEG_INFINITY], 2),
            Err(VectorError::NonFinite { position: 1 })
        );
    }

    #[test]
    fn accepts_finite_of_right_length() {
        assert_eq!(validate_vector(&[1.0, -0.0, 3.5], 3), Ok(()));
    }

    #[test]
    fn errors_display_helpfully() {
        let msg = VectorError::DimensionMismatch { expected: 8, actual: 3 }.to_string();
        assert!(msg.contains('8') && msg.contains('3'), "{msg}");
    }
}
