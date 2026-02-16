"""Shared shader sources inside the gl package."""

def get_default_shaders():
    """Return the default GLSL 330 vertex and fragment shader sources.

    These shaders are intended for desktop OpenGL (GLSL 330) use in the
    offline snapshot renderer. The returned strings contain full GLSL
    shader sources for vertex and fragment stages.

    Returns
    -------
    vertex_shader, fragment_shader : tuple[str, str]
        Tuple containing the vertex shader and fragment shader source code
        as plain strings.
    """
    vertex_shader = """

        #version 330

        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec3 aColor;

        out vec3 FragPos;
        out vec3 Normal;
        out vec3 Color;

        uniform mat4 transform;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main()
        {
          gl_Position = projection * view * model * transform * vec4(aPos, 1.0f);
          FragPos = vec3(model * transform * vec4(aPos, 1.0));
          // normal matrix should be computed outside and passed!
          Normal = mat3(transpose(inverse(view * model * transform))) * aNormal;
          Color = aColor;
        }

    """

    fragment_shader = """
        #version 330

        in vec3 Normal;
        in vec3 FragPos;
        in vec3 Color;

        out vec4 FragColor;

        uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
        uniform bool doSpecular = true;
        uniform float ambientStrength = 0.0;

        void main()
        {
          // ambient
          vec3 ambient = ambientStrength * lightColor;

          // diffuse
          vec3 norm = normalize(Normal);
          vec4 diffweights = vec4(0.6, 0.4, 0.4, 0.3);

          // key light (overhead)
          vec3 lightPos1 = vec3(0.0,5.0,5.0);
          vec3 lightDir = normalize(lightPos1 - FragPos);
          float diff = max(dot(norm, lightDir), 0.0);
          vec3 diffuse = diffweights[0]  * diff * lightColor;

          // headlight (at camera)
          vec3 lightPos2 = vec3(0.0,0.0,5.0);
          lightDir = normalize(lightPos2 - FragPos);
          vec3 ohlightDir = lightDir;
          diff = max(dot(norm, lightDir), 0.0);
          diffuse = diffuse + diffweights[1]  * diff * lightColor;

          // fill light (from below)
          vec3 lightPos3 = vec3(0.0,-5.0,5.0);
          lightDir = normalize(lightPos3 - FragPos);
          diff = max(dot(norm, lightDir), 0.0);
          diffuse = diffuse + diffweights[2] * diff * lightColor;

          // left right back lights
          vec3 lightPos4 = vec3(5.0,0.0,-5.0);
          lightDir = normalize(lightPos4 - FragPos);
          diff = max(dot(norm, lightDir), 0.0);
          diffuse = diffuse + diffweights[3] * diff * lightColor;

          vec3 lightPos5 = vec3(-5.0,0.0,-5.0);
          lightDir = normalize(lightPos5 - FragPos);
          diff = max(dot(norm, lightDir), 0.0);
          diffuse = diffuse + diffweights[3] * diff * lightColor;

          // specular
          vec3 result;
          if (doSpecular)
          {
            float specularStrength = 0.5;
            vec3 viewDir = normalize(-FragPos);
            vec3 reflectDir = reflect(ohlightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * lightColor;
            result = (ambient + diffuse + specular) * Color;
          }
          else
          {
            result = (ambient + diffuse) * Color;
          }
          FragColor = vec4(result, 1.0);
        }

    """

    return vertex_shader, fragment_shader


def get_webgl_shaders():
    """Return vertex and fragment shader source strings suitable for WebGL/Three.js.

    These shader snippets are small GLSL pieces that expect Three.js to
    provide built-in attributes/uniforms (e.g. projectionMatrix,
    modelViewMatrix, normalMatrix). They are used by the Jupyter
    pythreejs-based viewer.

    Returns
    -------
    vertex_shader, fragment_shader : tuple[str, str]
        Vertex and fragment shader source strings for WebGL / Three.js.
    """

    # Only declare custom attributes - Three.js provides position, normal, matrices
    # Don't declare position, normal, *Matrix
    # Only attributes like color , or uniforms like lightColor, ambientStrenght
    # Use normalMatrix instead of computing transpose...
    vertex_shader = """
        attribute vec3 color;
    
        varying vec3 vFragPos;
        varying vec3 vNormal;
        varying vec3 vColor;
    
        void main()
        {
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          vFragPos = vec3(modelViewMatrix * vec4(position, 1.0));
          vNormal = normalMatrix * normal;
          vColor = color;
        }
    """

    fragment_shader = """
        precision highp float;
    
        varying vec3 vNormal;
        varying vec3 vFragPos;
        varying vec3 vColor;
    
        uniform vec3 lightColor;
        uniform float ambientStrength;
    
        void main()
        {
          // ambient
          vec3 ambient = ambientStrength * lightColor;
    
          // diffuse
          vec3 norm = normalize(vNormal);
          vec4 diffweights = vec4(0.6, 0.4, 0.4, 0.3);
    
          // key light (overhead)
          vec3 lightPos1 = vec3(0.0, 5.0, 5.0);
          vec3 lightDir = normalize(lightPos1 - vFragPos);
          float diff = max(dot(norm, lightDir), 0.0);
          vec3 diffuse = diffweights[0] * diff * lightColor;
    
          // headlight (at camera)
          vec3 lightPos2 = vec3(0.0, 0.0, 5.0);
          lightDir = normalize(lightPos2 - vFragPos);
          vec3 ohlightDir = lightDir;
          diff = max(dot(norm, lightDir), 0.0);
          diffuse = diffuse + diffweights[1] * diff * lightColor;
    
          // fill light (from below)
          vec3 lightPos3 = vec3(0.0, -5.0, 5.0);
          lightDir = normalize(lightPos3 - vFragPos);
          diff = max(dot(norm, lightDir), 0.0);
          diffuse = diffuse + diffweights[2] * diff * lightColor;
    
          // left right back lights
          vec3 lightPos4 = vec3(5.0, 0.0, -5.0);
          lightDir = normalize(lightPos4 - vFragPos);
          diff = max(dot(norm, lightDir), 0.0);
          diffuse = diffuse + diffweights[3] * diff * lightColor;
    
          vec3 lightPos5 = vec3(-5.0, 0.0, -5.0);
          lightDir = normalize(lightPos5 - vFragPos);
          diff = max(dot(norm, lightDir), 0.0);
          diffuse = diffuse + diffweights[3] * diff * lightColor;
    
          // specular
          float specularStrength = 0.5;
          vec3 viewDir = normalize(-vFragPos);
          vec3 reflectDir = reflect(-ohlightDir, norm);
          float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
          vec3 specular = specularStrength * spec * lightColor;
    
          vec3 result = (ambient + diffuse + specular) * vColor;
          gl_FragColor = vec4(result, 1.0);
        }
    """

    return vertex_shader, fragment_shader

