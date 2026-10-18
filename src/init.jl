# Append a hint to the MethodError raised when a public function gets a non-array input.
function input_hint(io, e, argtypes, _kwargs)
    f = e.f === Core.kwcall ? e.args[2] : e.f
    f in (sigma_clip, sigma_clip!, sigma_clip_mask, sigma_clip_mask!, sigma_clip_bounds, sigma_clipped_stats) || return
    (isempty(argtypes) || argtypes[1] <: AbstractArray) && return
    print(io, "\n$f requires an AbstractArray; pass collect(x) to use an iterator")
    return
end

__init__() = Base.Experimental.register_error_hint(input_hint, MethodError)
