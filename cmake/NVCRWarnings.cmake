function(nvcr_set_project_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
        if(NVCR_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(
            ${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
            $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
            $<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>
            $<$<COMPILE_LANGUAGE:CXX>:-Wold-style-cast>)
        if(NVCR_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
        endif()
        if(NVCR_ENABLE_SANITIZERS)
            target_compile_options(
                ${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endfunction()

