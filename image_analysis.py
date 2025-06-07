from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import os
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

# Clarifai configuration
PAT = os.getenv('CLARIFAI_API_KEY')
# Use Clarifai's official general model
USER_ID = 'clarifai'
APP_ID = 'main'
# Using the general model
MODEL_ID = 'aaa03c23b3724a16a56b629203edc62c'  # General model
MODEL_VERSION_ID = None  # Latest version

# Supported languages with their codes
SUPPORTED_LANGUAGES = {
    'english': 'en',
    'spanish': 'es',
    'french': 'fr',
    'german': 'de',
    'italian': 'it',
    'portuguese': 'pt',
    'dutch': 'nl',
    'russian': 'ru',
    'chinese': 'zh',
    'japanese': 'ja',
    'korean': 'ko',
    'arabic': 'ar',
    'hindi': 'hi',
    'bengali': 'bn',
    'telugu': 'te',
    'tamil': 'ta',
    'kannada': 'kn',
    'malayalam': 'ml'
}

def validate_base64(base64_string):
    """
    Validate and convert base64 string to bytes
    """
    try:
        # Remove potential data URI prefix
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Add padding if necessary
        padding = 4 - (len(base64_string) % 4)
        if padding != 4:
            base64_string += '=' * padding

        # Convert to bytes
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {str(e)}")

def analyze_animal_image(image_base64, language='english'):
    """
    Analyze an animal image using Clarifai's models
    Args:
        image_base64: Base64 encoded image string
        language: Language for the response (default: 'english')
    """
    try:
        # Validate language
        if language.lower() not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Supported languages: {', '.join(SUPPORTED_LANGUAGES.keys())}")

        # Get language code
        lang_code = SUPPORTED_LANGUAGES[language.lower()]

        # Validate and convert base64 to bytes
        image_bytes = validate_base64(image_base64)

        # Create gRPC channel
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)

        # Create metadata for authentication
        metadata = (('authorization', f'Key {PAT}'),)

        # Prepare the request with language specification
        post_model_outputs_request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(
                user_id=USER_ID,
                app_id=APP_ID
            ),
            model_id=MODEL_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=image_bytes
                        )
                    )
                )
            ],
            model=resources_pb2.Model(
                output_info=resources_pb2.OutputInfo(
                    output_config=resources_pb2.OutputConfig(
                        language=lang_code
                    )
                )
            )
        )

        # Make the request
        response = stub.PostModelOutputs(post_model_outputs_request, metadata=metadata)

        # Check response status
        if response.status.code != status_code_pb2.SUCCESS:
            error_msg = response.status.description
            if response.status.details:
                error_msg += f" Details: {response.status.details}"
            raise Exception(f"Request failed, status: {error_msg}")

        # Process results
        results = []
        if hasattr(response.outputs[0].data, 'concepts'):
            for concept in response.outputs[0].data.concepts:
                # Filter for relevant concepts with broader criteria
                if any(keyword in concept.name.lower() for keyword in 
                    ['animal', 'cattle', 'cow', 'buffalo', 'sheep', 'goat', 
                     'healthy', 'sick', 'wound', 'injury', 'swelling', 'infection',
                     'farm', 'livestock', 'veterinary', 'mammal', 'domestic']):
                    results.append({
                        'name': concept.name,
                        'confidence': round(concept.value * 100, 2)
                    })

            # If no animal-related concepts found, include top 5 concepts
            if not results:
                for concept in list(response.outputs[0].data.concepts)[:5]:
                    results.append({
                        'name': concept.name,
                        'confidence': round(concept.value * 100, 2)
                    })

        return {
            'success': True,
            'results': results,
            'language': language,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'results': None,
            'language': language,
            'error': str(e)
        }

def get_health_assessment(analysis_results, language='english'):
    """
    Convert Clarifai analysis results into a health assessment
    Args:
        analysis_results: Results from analyze_animal_image
        language: Language for the response (default: 'english')
    """
    if not analysis_results['success']:
        return {
            'condition': 'unknown',
            'confidence': 0,
            'needs_vet': True,
            'recommendation': f"Could not analyze image. Error: {analysis_results['error']}",
            'language': language
        }

    # Process the results
    health_indicators = {
        'healthy': 0,
        'sick': 0,
        'severity': 0
    }

    # Keywords for health assessment (you can add translations for these keywords)
    health_keywords = {
        'healthy': ['healthy', 'normal', 'good', 'active', 'alert', 'standing', 'grazing'],
        'unhealthy': ['sick', 'ill', 'disease', 'infection', 'injury', 'wound', 'swelling', 'limp', 'lying'],
        'severe': ['severe', 'critical', 'acute', 'emergency', 'bleeding', 'collapsed']
    }

    # First, check if we detected an animal
    animal_detected = any(
        any(animal in result['name'].lower() for animal in 
            ['animal', 'cattle', 'cow', 'buffalo', 'sheep', 'goat', 'livestock', 'mammal', 'farm'])
        for result in analysis_results['results']
    )

    if not animal_detected:
        return {
            'condition': 'unknown',
            'confidence': 0,
            'needs_vet': False,
            'recommendation': "No animal detected in the image. Please ensure the animal is clearly visible in the photo.",
            'language': language
        }

    # Analyze each detected feature
    for result in analysis_results['results']:
        name = result['name'].lower()
        confidence = result['confidence']

        # Check health indicators
        if any(word in name for word in health_keywords['healthy']):
            health_indicators['healthy'] += confidence
        if any(word in name for word in health_keywords['unhealthy']):
            health_indicators['sick'] += confidence
        if any(word in name for word in health_keywords['severe']):
            health_indicators['severity'] += confidence

    # Make assessment
    assessment = {
        'language': language
    }
    
    if health_indicators['sick'] > 30:  # If sickness confidence is above 30%
        needs_vet = health_indicators['severity'] > 20
        assessment.update({
            'condition': 'potential health issue detected',
            'confidence': round(health_indicators['sick']),
            'needs_vet': needs_vet,
            'recommendation': (
                "Immediate veterinary attention recommended. "
                if needs_vet else 
                "Monitor closely and consult a vet if condition worsens."
            )
        })
    elif health_indicators['healthy'] > 50:
        assessment.update({
            'condition': 'appears healthy',
            'confidence': round(health_indicators['healthy']),
            'needs_vet': False,
            'recommendation': "No immediate health concerns detected. Continue regular monitoring."
        })
    else:
        assessment.update({
            'condition': 'unclear',
            'confidence': max(30, round((health_indicators['healthy'] + health_indicators['sick'])/2)),
            'needs_vet': True,
            'recommendation': "Unable to make a clear assessment. Recommend veterinary consultation for proper evaluation."
        })
    
    return assessment 